"""Paired common-checkpoint continuations with measured finite-response terms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.round31_prospective_20261005.worker import logits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    return {name: value.to(device='cuda', dtype=torch.float64).requires_grad_()
            for name, value in torch.load(path, map_location='cpu', weights_only=True).items()}


def clear(parameters):
    for value in parameters.values():
        value.grad = None


def gradient(parameters, tokens):
    clear(parameters)
    total = 0.
    for ids in tokens.split(1):
        lp = logits(parameters, ids).log_softmax(-1)
        loss = -lp.gather(-1, ids[:, 33:, None]).squeeze(-1).mean()
        (loss / len(tokens)).backward()
        total += float(loss.detach()) / len(tokens)
    return total


def quantities(z0, zt, target):
    lp0, lpt = z0.log_softmax(-1), zt.log_softmax(-1)
    p0 = lp0.exp()
    dz = zt - z0
    mean = (p0 * dz).sum(-1)
    difficulty = -lp0.gather(-1, target[..., None]).squeeze(-1)
    loss = -lpt.gather(-1, target[..., None]).squeeze(-1)
    work = mean - dz.gather(-1, target[..., None]).squeeze(-1)
    cost = (p0 * (lp0 - lpt)).sum(-1)
    q2 = .5 * (p0 * (dz - mean[..., None]).square()).sum(-1)
    identity = float((loss - difficulty - work - cost).abs().max())
    assert identity < 2e-10, identity
    assert float(cost.min()) > -2e-12
    return dict(loss=loss, difficulty=difficulty, work=work, cost=cost,
                quadratic=q2), identity


@torch.no_grad()
def parameter_dot(initial, current, derivative=None):
    derivative = derivative or {name: value.grad for name, value in initial.items()}
    return sum(float((derivative[name] * (current[name] - value)).sum())
               for name, value in initial.items())


def measure(initial, current, tokens, need_gradient=True, fixture=False):
    documents, saved = {}, {}
    maxerror = 0.
    if need_gradient:
        clear(initial)
    for index, ids in enumerate(tokens.split(1)):
        with torch.no_grad():
            zt = logits(current, ids)
        with torch.set_grad_enabled(need_gradient):
            z0 = logits(initial, ids)
            ce0 = -z0.log_softmax(-1).gather(-1, ids[:, 33:, None]).squeeze(-1).mean()
            if need_gradient:
                (ce0 / len(tokens)).backward()
        with torch.no_grad():
            terms, error = quantities(z0.detach(), zt, ids[:, 33:])
            maxerror = max(maxerror, error)
            for name, value in terms.items():
                documents.setdefault(name, []).append(float(value.mean()))
            if fixture and index == 0:
                saved = dict(before=z0[0, :8].detach().cpu().numpy(),
                             after=zt[0, :8].cpu().numpy(),
                             target=ids[0, 33:41].cpu().numpy())
    row = {name: float(np.mean(value)) for name, value in documents.items()}
    row['identity_maxerror'] = maxerror
    if need_gradient:
        row['parameter_work'] = parameter_dot(initial, current)
        row['representation_remainder'] = row['work'] - row['parameter_work']
        row['softmax_remainder'] = row['cost'] - row['quadratic']
        row['epsilon'] = row['loss'] - row['difficulty'] - row['parameter_work'] - row['quadratic']
        assert abs(row['epsilon'] - row['representation_remainder'] - row['softmax_remainder']) < 2e-10
    assert all(np.isfinite(value) for value in row.values())
    return row, documents, saved


def make_optimizer(parameters, method, rate):
    chosen = [parameters['embed_out.weight']] if method == 'head_low' else list(parameters.values())
    if method == 'adam_low':
        return torch.optim.Adam(chosen, lr=rate, betas=(.9, .95), eps=1e-8, foreach=False)
    if method.startswith('momentum'):
        return torch.optim.SGD(chosen, lr=.1 * rate, momentum=.9, foreach=False)
    return torch.optim.SGD(chosen, lr=rate, foreach=False)


@torch.no_grad()
def step_norm(initial, current):
    return sum(float((current[name] - value).square().sum())
               for name, value in initial.items()) ** .5


def inputs(args):
    root = Path(args.root)
    plan = json.loads((root / 'plan.json').read_text())
    assert sha(__file__) == plan['producer_sha256']
    assert sha(Path(__file__).parents[1] / 'round31_prospective_20261005/worker.py') == plan['forward_sha256']
    assert sha(root / 'weights/143000.pt') in [plan['checkpoint']['sha256'], plan['checkpoint'].get('remote_serialization_sha256')]
    data = Path(args.data_root)
    assert sha(data / 'new-tokens.npy') == plan['tokens_sha256']
    assert sha(data / 'new-order.npy') == plan['order_sha256']
    raw = np.load(data / 'new-tokens.npy')[np.load(data / 'new-order.npy')]
    tokens = torch.tensor(raw.astype(np.int64), device='cuda')
    return root, plan, tokens


def calibrate(args):
    root, plan, tokens = inputs(args)
    out = root / 'calibration.json'
    assert not out.exists()
    initial = load(root / 'weights/143000.pt')
    calibration = tokens[4096:4112]
    gradient(initial, calibration)
    frozen_gradient = {name: value.grad.detach().clone() for name, value in initial.items()}
    rows = []
    target_q2 = None
    rate = 1e-6
    for method in ['sgd_low', 'adam_low', 'adam_low', 'adam_low']:
        current = {name: value.detach().clone().requires_grad_() for name, value in initial.items()}
        trial_rate = plan['rates']['sgd_low'] if method == 'sgd_low' else rate
        optimizer = make_optimizer(current, method, trial_rate)
        for name, value in current.items():
            value.grad = frozen_gradient[name].clone()
        optimizer.step()
        row, _, _ = measure(initial, current, calibration, need_gradient=False)
        rows.append(dict(method=method, rate=trial_rate, update_norm=step_norm(initial, current), **row))
        if method == 'sgd_low':
            target_q2 = row['quadratic']
        else:
            if abs(row['quadratic'] / target_q2 - 1.) < .005 or len(rows) == 4:
                break
            rate *= (target_q2 / row['quadratic']) ** .5
        del current, optimizer
    assert abs(rows[-1]['quadratic'] / target_q2 - 1.) < .03
    write(out, dict(completed=True, method='reset one-step Q2 scale, first16 calibration documents only',
                    adam_rate=rows[-1]['rate'], target_Q2=target_q2, rows=rows,
                    plan_sha256=sha(root / 'plan.json'), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print(json.dumps(dict(completed=True, adam_rate=rows[-1]['rate'], Q2=target_q2)), flush=True)


def final_tokens(initial, current, probe, derivative, out):
    saved = {}
    displacement = {name: current[name].detach() - value.detach() for name, value in initial.items()}
    frozen = {name: value.detach() for name, value in initial.items()}
    for ids in probe.split(1):
        with torch.no_grad():
            z0, linear = torch.func.jvp(lambda p: logits(p, ids), (frozen,), (displacement,))
            zt = logits(current, ids)
            terms, _ = quantities(z0, zt, ids[:, 33:])
            p0 = z0.softmax(-1)
            a = (p0 * linear).sum(-1) - linear.gather(-1, ids[:, 33:, None]).squeeze(-1)
            terms['parameter_work'] = a
            terms['epsilon'] = terms['loss'] - terms['difficulty'] - a - terms['quadratic']
            terms['representation_remainder'] = terms['work'] - a
            for name, value in terms.items():
                saved.setdefault(name, []).append(value.cpu().numpy())
    arrays = {name: np.concatenate(values) for name, values in saved.items()}
    parity = abs(float(arrays['parameter_work'].mean()) - parameter_dot(initial, current, derivative))
    assert parity < 2e-9, parity
    np.savez_compressed(out / 'final-tokens.npz', **arrays)
    return parity


def run(args):
    root, plan, tokens = inputs(args)
    out = root / (f'pilot-{args.method}' if args.pilot else f'r{args.replica}-{args.method}')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    initial = load(root / 'weights/143000.pt')
    current = {name: value.detach().clone().requires_grad_() for name, value in initial.items()}
    train, probe = tokens[:4096], tokens[4160:4192]
    seed = plan['permutation_seeds'][args.replica]
    permutation = np.random.default_rng(seed).permutation(4096)
    rate = (json.loads((root / 'calibration.json').read_text())['adam_rate']
            if args.method == 'adam_low' else plan['rates'][args.method])
    optimizer = make_optimizer(current, args.method, rate)
    updates = 64 if args.pilot else plan['updates']
    write(out / 'registration.json', dict(method=args.method, replica=args.replica, seed=seed,
          permutation=permutation.tolist(), rate=rate, updates=updates, warmup=plan['warmup'],
          plan_sha256=sha(root / 'plan.json'), calibration_sha256=sha(root / 'calibration.json'),
          gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(), dtype='float64'))
    gradient(initial, probe)
    probe_gradient = {name: value.grad.detach().clone() for name, value in initial.items()}
    probe_rows, rows, documents = [], [], {}
    readouts = [value for value in plan['readouts'] if value <= updates]
    if updates not in readouts:
        readouts.append(updates)
    identity_error = 0.
    for step in range(updates + 1):
        if step in readouts:
            fixed, _, fixture = measure(initial, current, probe, need_gradient=False, fixture=step == updates)
            fixed['parameter_work'] = parameter_dot(initial, current, probe_gradient)
            fixed['representation_remainder'] = fixed['work'] - fixed['parameter_work']
            fixed['softmax_remainder'] = fixed['cost'] - fixed['quadratic']
            fixed['epsilon'] = fixed['loss'] - fixed['difficulty'] - fixed['parameter_work'] - fixed['quadratic']
            probe_rows.append(dict(step=step, **fixed))
            write(out / 'fixed-probe.json', probe_rows)
            if fixture:
                np.savez_compressed(out / 'fixture.npz', **fixture)
        if step == updates:
            break
        ids = train[permutation[step * 8:(step + 1) * 8]]
        row, docs, _ = measure(initial, current, ids)
        identity_error = max(identity_error, row['identity_maxerror'])
        factor = min((step + 1) / plan['warmup'], 1.)
        raw_rate = rate * factor * (.1 if args.method.startswith('momentum') else 1.)
        for group in optimizer.param_groups:
            group['lr'] = raw_rate
        row.update(step=step, next_raw_lr=raw_rate,
                   next_effective_lr=rate * factor if args.method != 'adam_low' else None,
                   displacement_norm=step_norm(initial, current))
        if step == 0:
            assert row['parameter_work'] == row['quadratic'] == row['epsilon'] == 0.
        rows.append(row)
        for name, value in docs.items():
            documents.setdefault(name, []).append(value)
        if args.method == 'head_low':
            for name, value in current.items():
                value.requires_grad_(name == 'embed_out.weight')
        gradient(current, ids)
        optimizer.step()
        if (step + 1) % 32 == 0:
            write(out / 'trace.json', rows)
            np.savez_compressed(out / 'document.npz', **{name: np.asarray(value) for name, value in documents.items()})
            progress = dict(step=step + 1, total=updates, seconds=time.monotonic() - started,
                            method=args.method, replica=args.replica, identity_maxerror=identity_error)
            write(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            assert progress['seconds'] < plan['job_budget_seconds'], 'Registered job budget exceeded'
    parity = None if args.pilot else final_tokens(initial, current, probe, probe_gradient, out)
    if not args.pilot:
        torch.save({name: value.detach().cpu() for name, value in current.items()}, out / 'final.pt')
    write(out / 'summary.json', dict(completed=True, method=args.method, replica=args.replica,
          updates=updates, seconds=time.monotonic() - started, identity_maxerror=identity_error,
          token_A_gradient_dot_maxerror=parity, displacement_norm=step_norm(initial, current),
          gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__,
          max_memory_bytes=torch.cuda.max_memory_allocated(), plan_sha256=sha(root / 'plan.json'),
          producer_sha256=sha(__file__), scope='Common checkpoint, shared arrival stream, observed finite response'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['calibrate', 'run'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--method', default='sgd_low', choices=['sgd_low', 'momentum_low', 'adam_low', 'sgd_high', 'momentum_high', 'head_low'])
    parser.add_argument('--replica', type=int, default=0)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == 'calibrate':
        calibrate(args)
    else:
        run(args)
