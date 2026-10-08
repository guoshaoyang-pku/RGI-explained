"""Prospective Pythia SGD, native-metric sharpening and full fixed tangent."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def logits(p, ids, start=32):
    batch, length = ids.shape
    inv = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, device=ids.device).float() / 16))
    angles = torch.arange(length, device=ids.device).float()[:, None] * inv[None]
    angles = torch.cat([angles, angles], -1)
    cosine = angles.cos().to(p['gpt_neox.embed_in.weight'].dtype)
    sine = angles.sin().to(p['gpt_neox.embed_in.weight'].dtype)

    def rotate(x):
        a, b = x[..., :8], x[..., 8:16]
        part = x[..., :16] * cosine + torch.cat([-b, a], -1) * sine
        return torch.cat([part, x[..., 16:]], -1)

    mask = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
    hidden = F.embedding(ids, p['gpt_neox.embed_in.weight'])
    for layer in range(6):
        stem = 'gpt_neox.layers.' + str(layer) + '.'
        norm = F.layer_norm(hidden, (512,), p[stem + 'input_layernorm.weight'], p[stem + 'input_layernorm.bias'], 1e-5)
        fused = F.linear(norm, p[stem + 'attention.query_key_value.weight'], p[stem + 'attention.query_key_value.bias'])
        q, k, v = fused.view(batch, length, 8, 192).transpose(1, 2).chunk(3, -1)
        score = (rotate(q) @ rotate(k).transpose(-1, -2)) / 8
        gate = score.masked_fill(mask, -torch.inf).softmax(-1)
        mixed = (gate @ v).transpose(1, 2).reshape(batch, length, 512)
        attention = F.linear(mixed, p[stem + 'attention.dense.weight'], p[stem + 'attention.dense.bias'])
        norm = F.layer_norm(hidden, (512,), p[stem + 'post_attention_layernorm.weight'], p[stem + 'post_attention_layernorm.bias'], 1e-5)
        mlp = F.gelu(F.linear(norm, p[stem + 'mlp.dense_h_to_4h.weight'], p[stem + 'mlp.dense_h_to_4h.bias']), approximate='none')
        mlp = F.linear(mlp, p[stem + 'mlp.dense_4h_to_h.weight'], p[stem + 'mlp.dense_4h_to_h.bias'])
        hidden = mlp + attention + hidden
    hidden = F.layer_norm(hidden, (512,), p['gpt_neox.final_layer_norm.weight'], p['gpt_neox.final_layer_norm.bias'], 1e-5)
    return F.linear(hidden[:, start:-1], p['embed_out.weight'])


def loss(y, ids, start=32):
    return F.cross_entropy(y.flatten(0, 1), ids[:, start + 1:].flatten(), reduction='none').reshape(len(ids), -1).mean(-1)


def load(path):
    return {n: v.cuda().double() for n, v in torch.load(path, map_location='cpu', weights_only=True).items()}


def tangent(base, delta, ids):
    y0, dy = torch.func.jvp(lambda p: logits(p, ids), (base,), (delta,))
    return y0 + dy


@torch.no_grad()
def radial_step(p, grad, eta, heads, output_head=False):
    coefficients, total_norm, projected_norm = [], 0., 0.
    for layer, head in heads:
        stem = 'gpt_neox.layers.' + str(layer) + '.attention.query_key_value.'
        slices = [(stem + suffix, slice(192 * head, 192 * head + 128)) for suffix in ['weight', 'bias']]
        norm = sum(p[n][s].square().sum() for n, s in slices)
        dot = sum((p[n][s] * grad[n][s]).sum() for n, s in slices)
        coefficient = dot / norm
        factor = 1 - eta * coefficient
        assert float(factor) > 0, 'radial Euler crossed zero'
        for n, s in slices:
            p[n][s].mul_(factor)
        coefficients.append(float(coefficient))
        projected_norm += float(dot.square() / norm)
    if output_head:
        n = 'embed_out.weight'
        norm = p[n].square().sum()
        dot = (p[n] * grad[n]).sum()
        coefficient = dot / norm
        factor = 1 - eta * coefficient
        assert float(factor) > 0
        p[n].mul_(factor)
        coefficients.append(float(coefficient))
        projected_norm += float(dot.square() / norm)
    total_norm = sum(float(g.square().sum()) for g in grad.values())
    return coefficients, projected_norm, total_norm


def gradients(p, ids, base=None, denominator=None):
    for v in p.values():
        v.grad = None
    y = tangent(base, p, ids) if base is not None else logits(p, ids)
    (loss(y, ids).sum() / (denominator or len(ids))).backward()
    return {n: v.grad for n, v in p.items()}


def freeze(args):
    root = Path(args.root)
    plan = json.loads((root / 'calibration-plan.json').read_text())
    assert plan['worker_sha256'] == sha(__file__)
    for entry in plan['files']:
        assert sha(root / entry['path']) == entry['sha256']
    base = load(root / 'weights' / '32000.pt')
    p = {n: v.clone().requires_grad_() for n, v in base.items()}
    raw = np.load(root / 'tokens.npy')[np.load(root / 'order.npy')]
    calibration = torch.tensor(raw[4096:4160].astype(np.int64), device='cuda')
    g = {n: torch.zeros_like(v) for n, v in p.items()}
    for ids in calibration.split(4):
        micro = gradients(p, ids, denominator=64)
        with torch.no_grad():
            for n in g:
                g[n].add_(micro[n])
    rows = []
    with torch.no_grad():
        for layer in range(6):
            for head in range(8):
                stem = 'gpt_neox.layers.' + str(layer) + '.attention.query_key_value.'
                names = [stem + suffix for suffix in ['weight', 'bias']]
                s = slice(192*head, 192*head+128)
                norm = sum(p[n][s].square().sum() for n in names)
                dot = sum((p[n][s]*g[n][s]).sum() for n in names)
                rows.append(dict(head=[layer,head], A=-float(dot)/2, norm_squared=float(norm)))
        n = 'embed_out.weight'
        rows.append(dict(head='LM', A=-float((p[n]*g[n]).sum()), norm_squared=float(p[n].square().sum())))
    write(root / 'frozen-boundary.json', dict(completed=True, calibration_plan_sha256=sha(root/'calibration-plan.json'),
          coefficients=rows, calibration_documents=64, formula='For QK: r(t)^2=1+8*A*t/norm0; LM: r(t)^3=1+3*A*t/norm0. predicted_gain=sum A*(1-1/r).',
          scope='Fixed natural calibration derivative, separable inverse-scale closure, actual Euclidean radial metric; no teacher labels or positive coefficient guaranteed.'))
    torch.save({n:v.cpu() for n,v in g.items()}, root/'calibration-gradient.pt')


def engineering(args):
    started = time.monotonic()
    p = load(args.weights)
    generator = torch.Generator().manual_seed(319999)
    ids = torch.randint(50304, (1, 40), generator=generator).cuda()
    from experiments.round26_pythia_temporal_20261005.verify import Decoder
    model = Decoder().cuda().double()
    model.load_state_dict(p)
    with torch.no_grad():
        reference = model(ids)[:, 32:-1]
        direct = logits(p, ids)
    parity = float((reference - direct).abs().max())
    assert parity < 2e-10, parity
    del model
    parameters = {n: v.clone().requires_grad_() for n, v in p.items()}
    delta = {n: torch.zeros_like(v).requires_grad_() for n, v in p.items()}
    g = gradients(parameters, ids)
    gt = gradients(delta, ids, base=p)
    ge = max(float((g[n] - gt[n]).abs().max()) for n in p)
    assert ge < 2e-10, ge
    with torch.no_grad():
        direction = {n: -0.001 * g[n] for n in p}
        jvp = torch.func.jvp(lambda pp: logits(pp, ids), (p,), (direction,))[1]
        epsilon = 0.00001
        plus = logits({n: p[n] + epsilon * direction[n] for n in p}, ids)
        minus = logits({n: p[n] - epsilon * direction[n] for n in p}, ids)
        fde = float(((plus - minus) / (2 * epsilon) - jvp).abs().max())
        assert fde < 2e-7, fde
        update = {n: v.clone() for n, v in p.items()}
        radial_step(update, g, .001, [(0, 0), (5, 7)], True)
        projection_identity = sum(float((g[n] * (p[n] - update[n]) / .001).sum()) for n in p)
        update_norm = sum(float(((p[n] - update[n]) / .001).square().sum()) for n in p)
        assert abs(projection_identity - update_norm) < 2e-9
    del parameters, delta, g, gt, direction, update
    ids = torch.randint(50304, (4, 256), generator=generator).cuda()
    timings = {}
    for kind in ['full', 'tangent']:
        parameters = {n: (torch.zeros_like(v) if kind == 'tangent' else v.clone()).requires_grad_() for n, v in p.items()}
        torch.cuda.synchronize()
        begin = time.monotonic()
        gradients(parameters, ids, base=p if kind == 'tangent' else None)
        torch.cuda.synchronize()
        timings[kind] = time.monotonic() - begin
        del parameters
    write(args.out, dict(completed=True, synthetic_only=True, code_sha256=sha(__file__), weight_sha256=sha(args.weights),
          output_parity_maxerror=parity, anchor_gradient_maxerror=ge, jvp_finite_difference_maxerror=fde,
          projection_identity_error=abs(projection_identity-update_norm), timings_batch4_256=timings,
          max_cuda_memory_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started))


@torch.no_grad()
def evaluate(base, p, tokens, kind, microbatch):
    losses, energy = [], []
    for ids in tokens.split(microbatch):
        y0 = logits(base, ids)
        y = tangent(base, p, ids) if kind == 'tangent' else logits(p, ids)
        prob0, prob = y0.softmax(-1), y.softmax(-1)
        losses.extend(loss(y, ids).cpu().tolist())
        energy.extend((prob-prob0).square().sum(-1).mean(-1).cpu().tolist())
    return np.asarray(losses), np.asarray(energy)


def run(args):
    started = time.monotonic()
    root = Path(args.root)
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['worker_sha256'] == sha(__file__)
    for entry in plan['files']:
        assert sha(root / entry['path']) == entry['sha256'], entry['path']
    spec = plan['science']
    base = load(root / 'weights' / '32000.pt')
    p = {n: (torch.zeros_like(v) if args.kind == 'tangent' else v.clone()).requires_grad_() for n, v in base.items()}
    raw = np.load(root / 'tokens.npy')[np.load(root / 'order.npy')]
    train = torch.tensor(raw[:spec['train_documents']].astype(np.int64), device='cuda')
    evaluation = torch.tensor(raw[spec['train_documents']+spec['calibration_documents']:].astype(np.int64), device='cuda')
    permutation = np.random.default_rng(spec['stream_seed'] + args.replica).permutation(len(train))
    selected = json.loads((root / 'selection.json').read_text())['heads']
    heads = [(l, h) for l in range(6) for h in range(8)] if args.kind == 'radial_all' else selected
    half = args.kind == 'half'
    eta = spec['eta'] / (2 if half else 1)
    out = root / ('r' + str(args.replica) + '-' + args.kind)
    out.mkdir(exist_ok=False)
    write(out / 'registration.json', dict(plan_sha256=sha(root/'plan.json'), kind=args.kind, replica=args.replica,
          gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), eta=eta, permutation=permutation.tolist(), heads=heads,
          torch_version=torch.__version__, code_sha256=sha(__file__)))
    arrays = {}
    l, e = evaluate(base, p, evaluation, args.kind, spec['microbatch'])
    arrays['loss/0'], arrays['energy/0'] = l, e
    trace = []
    first = None
    for index in range(spec['updates']):
        batch = train[permutation[index*spec['batch_size']:(index+1)*spec['batch_size']]]
        for substep in range(2 if half else 1):
            accumulated = {n: torch.zeros_like(v) for n, v in p.items()}
            for ids in batch.split(spec['microbatch']):
                g = gradients(p, ids, base=base if args.kind == 'tangent' else None, denominator=spec['batch_size'])
                with torch.no_grad():
                    for n in p:
                        accumulated[n].add_(g[n])
            if index == 0 and substep == 0:
                first = dict(ids=batch.cpu(), gradient={n: g.cpu() for n, g in accumulated.items()}, denominator=spec['batch_size'])
            with torch.no_grad():
                if args.kind.startswith('radial'):
                    c, projected, total = radial_step(p, accumulated, eta, heads, args.kind == 'radial_all')
                    trace.append(dict(update=index+1, coefficients=c, projected_squared_norm=projected, total_squared_norm=total))
                else:
                    for n in p:
                        p[n].sub_(accumulated[n], alpha=eta)
            del accumulated
        if index == 0:
            first['after'] = {n: v.detach().cpu() for n, v in p.items()}
            torch.save(first, out / 'capture.pt')
            del first
        if index + 1 in spec['readouts']:
            torch.save({n: v.detach().cpu() for n, v in p.items()}, out / ('state-' + str(index+1) + '.pt'))
            l, e = evaluate(base, p, evaluation, args.kind, spec['microbatch'])
            arrays['loss/' + str(index+1)], arrays['energy/' + str(index+1)] = l, e
            np.savez_compressed(out / 'readouts.npz', **arrays)
            print(json.dumps(dict(kind=args.kind, replica=args.replica, update=index+1, mean_ce=float(l.mean()), seconds=time.monotonic()-started)), flush=True)
        if time.monotonic()-started > plan['budget_seconds']:
            raise TimeoutError('registered continuation budget')
    write(out / 'trace.json', trace)
    write(out / 'summary.json', dict(completed=True, kind=args.kind, replica=args.replica, seconds=time.monotonic()-started,
          plan_sha256=sha(root/'plan.json'), code_sha256=sha(__file__), max_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
          artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file()}))


@torch.no_grad()
def compare(args):
    root = Path(args.root)
    plan = json.loads((root/'plan.json').read_text())
    assert plan['worker_sha256'] == sha(__file__)
    spec = plan['science']
    base = load(root/'weights'/'32000.pt')
    raw = np.load(root/'tokens.npy')[np.load(root/'order.npy')]
    evaluation = torch.tensor(raw[spec['train_documents']+spec['calibration_documents']:].astype(np.int64), device='cuda')
    kinds = ['full','half','radial_cold','radial_all','tangent']
    states = {}
    for kind in kinds:
        folder=root/('r'+str(args.replica)+'-'+kind)
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['completed'] and summary['plan_sha256']==sha(root/'plan.json')
        for step in spec['readouts']:
            path=folder/('state-'+str(step)+'.pt')
            assert sha(path)==summary['artifacts'][path.name]
            states[kind,step]=load(path)
    arrays = {}
    for ids in evaluation.split(spec['microbatch']):
        anchor = logits(base,ids).softmax(-1)
        probabilities={}
        for (kind,step),state in states.items():
            y=tangent(base,state,ids) if kind=='tangent' else logits(state,ids)
            probabilities[kind,step]=y.softmax(-1)
        for start,end in spec['windows']:
            prefix=str(start)+'-'+str(end)+'/'
            full=probabilities['full',end]-(anchor if start==0 else probabilities['full',start])
            energy=full.square().sum(-1).mean(-1).cpu().numpy()
            arrays.setdefault(prefix+'full_energy',[]).append(energy)
            for kind in kinds[1:]:
                motion=probabilities[kind,end]-(anchor if start==0 else probabilities[kind,start])
                error=(motion-full).square().sum(-1).mean(-1).cpu().numpy()
                arrays.setdefault(prefix+kind+'_error_energy',[]).append(error)
    dest=root/('comparison-'+str(args.replica)+'.npz')
    assert not dest.exists()
    np.savez_compressed(dest,**{n:np.concatenate(v) for n,v in arrays.items()})
    write(root/('comparison-'+str(args.replica)+'.json'),dict(completed=True,replica=args.replica,
          plan_sha256=sha(root/'plan.json'),raw_sha256=sha(dest),worker_sha256=sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--kind', choices=['full', 'half', 'radial_cold', 'radial_all', 'tangent'])
    parser.add_argument('--replica', type=int)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--weights')
    parser.add_argument('--out')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.engineering:
        engineering(args)
    elif args.freeze:
        freeze(args)
    elif args.compare:
        compare(args)
    else:
        run(args)
