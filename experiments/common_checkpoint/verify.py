"""Independently verify and summarize the registered common-checkpoint evidence."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


FIELDS = ['loss', 'difficulty', 'work', 'cost', 'quadratic']
TERMS = ['parameter_work', 'quadratic', 'epsilon']
REQUIRED = ['trace.json', 'document.npz', 'fixed-probe.json', 'fixture.npz',
            'final-tokens.npz', 'summary.json', 'registration.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_probability(z):
    z = np.asarray(z, dtype=np.float64)
    shifted = z - z.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def fixture_terms(z0, zt, target):
    l0, lt = log_probability(z0), log_probability(zt)
    p0 = np.exp(l0)
    dz = np.asarray(zt, dtype=np.float64) - np.asarray(z0, dtype=np.float64)
    mean = np.sum(p0 * dz, -1)
    return dict(loss=-np.take_along_axis(lt, target[..., None], -1)[..., 0],
                difficulty=-np.take_along_axis(l0, target[..., None], -1)[..., 0],
                work=mean-np.take_along_axis(dz, target[..., None], -1)[..., 0],
                cost=np.sum(p0 * (l0-lt), -1),
                quadratic=.5*np.sum(p0 * (dz-mean[..., None])**2, -1))


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64)**2)))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-14 else None


def response_stats(values, gate):
    m = values['loss']-values['difficulty']
    a, q2, eps = (values[key] for key in TERMS)
    variance = float(np.var(m))
    relative = ratio(rms(eps), rms(m))
    shares = {key: ratio(float(np.mean((values[key]-values[key].mean())*(m-m.mean()))), variance)
              for key in TERMS}
    dropped = {key: dict(absolute_rms=rms(m-(a+q2+eps-values[key])),
                         relative_rms=ratio(rms(m-(a+q2+eps-values[key])), rms(m)))
               for key in TERMS}
    total_sst = float(np.sum((values['loss']-values['loss'].mean())**2))
    frozen_fraction = 1-float(np.sum(m**2))/total_sst if total_sst > 1e-28 else None
    return dict(samples=int(m.size), loss_mean=float(values['loss'].mean()),
                loss_std=float(values['loss'].std()), difficulty_mean=float(values['difficulty'].mean()),
                difficulty_std=float(values['difficulty'].std()), response_mean=float(m.mean()),
                response_std=float(m.std()), response_rms=rms(m),
                parameter_work_mean=float(a.mean()), quadratic_mean=float(q2.mean()),
                epsilon_rms=rms(eps), response_relative_rms=relative,
                response_gate_passed=relative <= gate if relative is not None else None,
                difficulty_prediction_fraction=frozen_fraction,
                signed_covariance_shares=shares, drop_one=dropped,
                representation_remainder_rms=rms(values['representation_remainder']),
                softmax_remainder_rms=rms(values['softmax_remainder']),
                quadratic_vs_KL_relative_rms=ratio(rms(values['cost']-q2), rms(values['cost'])))


def window_stats(values, plan):
    result = {}
    for label, start, stop in [('full', 0, plan['updates']), ('near', 0, 128), ('far', plan['updates']-128, plan['updates'])]:
        selected = {key: value[start:stop] for key, value in values.items()}
        blocks = {key: value.reshape(-1, 64).mean(-1) for key, value in selected.items()}
        highpass = {key: value-np.repeat(blocks[key], 64) for key, value in selected.items()}
        result[label] = {mode: response_stats(data, plan['gates']['response_relative_rms'])
                         for mode, data in [('raw', selected), ('highpass64', highpass), ('blockmean64', blocks)]}
        result[label]['raw']['difficulty_gate_passed'] = (
            result[label]['raw']['difficulty_prediction_fraction'] >= plan['gates']['difficulty_prediction_fraction']
            if result[label]['raw']['difficulty_prediction_fraction'] is not None else None)
    return result


def gap_stats(values):
    mean, std = float(values.mean()), float(values.std())
    near_zero = abs(mean) <= 1e-8
    return dict(mean=mean, std=std, rms=rms(values), centered_rms=std,
                absolute_mean=abs(mean), mean_near_zero=near_zero,
                std_over_absolute_mean=ratio(std, abs(mean)) if not near_zero else None,
                constant_gap_gate_passed=std/abs(mean) <= .1 if not near_zero else None,
                zero_gap=rms(values) <= 1e-10)


def verify(root, data_root=None):
    plan = json.loads((root/'plan.json').read_text())
    plan_sha = sha(root/'plan.json')
    errors, failures, incomplete, manifest = {}, {}, [], []
    traces, token_arrays, probe_arrays, document_arrays, summaries, runs = {}, {}, {}, {}, {}, {}

    def check(label, actual, expected, tolerance=2e-10):
        left, right = np.asarray(actual), np.asarray(expected)
        if left.shape != right.shape:
            failures[label] = dict(actual_shape=list(left.shape), expected_shape=list(right.shape))
            return
        error = float(np.max(np.abs(left-right))) if left.size else 0.
        errors[label] = error
        if not np.isfinite(error) or error > tolerance:
            failures[label] = dict(error=error, tolerance=tolerance)

    def fact(label, value):
        check(label, int(bool(value)), 1, 0)

    def record(path):
        manifest.append(dict(path=str(path.relative_to(root)), sha256=sha(path)))

    def algebra(label, values, has_a=True):
        check(label+'/W_plus_Q', values['loss']-values['difficulty'], values['work']+values['cost'])
        for key in ['cost', 'quadratic']:
            check(label+'/nonnegative_'+key, np.minimum(values[key], 0), np.zeros_like(values[key]), 2e-12)
        if has_a:
            check(label+'/R', values['representation_remainder'], values['work']-values['parameter_work'])
            check(label+'/epsilon', values['epsilon'], values['loss']-values['difficulty']-values['parameter_work']-values['quadratic'])
            softmax = values['cost']-values['quadratic']
            check(label+'/epsilon_R_softmax', values['epsilon'], values['representation_remainder']+softmax)
            if 'softmax_remainder' in values:
                check(label+'/softmax_remainder', values['softmax_remainder'], softmax)

    def row_arrays(rows, fields):
        return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in fields}

    record(root/'plan.json')
    source = Path(__file__).with_name('worker.py')
    if source.exists():
        fact('producer_source_SHA', sha(source) == plan['producer_sha256'])
    forward = Path(__file__).parents[1]/'round31_prospective_20261005/worker.py'
    if forward.exists():
        fact('forward_source_SHA', sha(forward) == plan['forward_sha256'])
    calibration_path = root/'calibration.json'
    calibration = None
    if calibration_path.exists():
        record(calibration_path)
        calibration = json.loads(calibration_path.read_text())
        fact('calibration_completed', calibration['completed'])
        fact('calibration_plan_SHA', calibration['plan_sha256'] == plan_sha)
        check('calibration_target_Q2', calibration['target_Q2'], calibration['rows'][0]['quadratic'])
        fact('calibration_Q2_scale_gate', abs(calibration['rows'][-1]['quadratic']/calibration['target_Q2']-1) < .03)
        check('calibration_adam_rate', calibration['adam_rate'], calibration['rows'][-1]['rate'], 0)
        calibration_sha = sha(calibration_path)
    else:
        incomplete.append(dict(run='calibration', missing=['calibration.json']))
        calibration_sha = None
    raw_tokens = None
    if data_root is not None:
        for filename, key in [('new-tokens.npy', 'tokens_sha256'), ('new-order.npy', 'order_sha256')]:
            fact(filename+'/registered_SHA', sha(data_root/filename) == plan[key])
        raw_tokens = np.load(data_root/'new-tokens.npy')[np.load(data_root/'new-order.npy')]
        check('input_pool_shape', raw_tokens.shape, (4672, 256), 0)

    for replica, method in itertools.product(range(len(plan['permutation_seeds'])), plan['methods']):
        name = f'r{replica}-{method}'
        path = root/name
        missing = [filename for filename in REQUIRED if not (path/filename).exists()]
        if missing:
            incomplete.append(dict(run=name, missing=missing))
            continue
        try:
            summary = json.loads((path/'summary.json').read_text())
            registration = json.loads((path/'registration.json').read_text())
            rows = json.loads((path/'trace.json').read_text())
            fixed = json.loads((path/'fixed-probe.json').read_text())
            docs = dict(np.load(path/'document.npz'))
            fixture = dict(np.load(path/'fixture.npz'))
            tokens = dict(np.load(path/'final-tokens.npz'))
            for filename in REQUIRED:
                record(path/filename)
            fact(name+'/completed', summary['completed'])
            for record_label, metadata in [('summary', summary), ('registration', registration)]:
                fact(name+'/'+record_label+'_method', metadata['method'] == method)
                check(name+'/'+record_label+'_replica', metadata['replica'], replica, 0)
                check(name+'/'+record_label+'_updates', metadata['updates'], plan['updates'], 0)
                fact(name+'/'+record_label+'_plan_SHA', metadata['plan_sha256'] == plan_sha)
            fact(name+'/recorded_source_SHA', summary['producer_sha256'] == plan['producer_sha256'])
            fact(name+'/calibration_SHA', registration['calibration_sha256'] == calibration_sha)
            fact(name+'/dtype', registration['dtype'] == 'float64')
            check(name+'/warmup', registration['warmup'], plan['warmup'], 0)
            check(name+'/seed', registration['seed'], plan['permutation_seeds'][replica], 0)
            check(name+'/permutation', registration['permutation'],
                  np.random.default_rng(plan['permutation_seeds'][replica]).permutation(4096), 0)
            if method != 'adam_low' or calibration is not None:
                rate = calibration['adam_rate'] if method == 'adam_low' else plan['rates'][method]
                check(name+'/registered_rate', registration['rate'], rate, 0)
                scale = .1 if method.startswith('momentum') else 1.
                expected_rate = rate*np.minimum((np.arange(plan['updates'])+1)/plan['warmup'], 1.)*scale
                check(name+'/warmup_rates', [row['next_raw_lr'] for row in rows], expected_rate, 1e-18)
            check(name+'/step_schedule', [row['step'] for row in rows], np.arange(plan['updates']), 0)
            check(name+'/readout_schedule', [row['step'] for row in fixed], plan['readouts'], 0)
            fields = FIELDS+['parameter_work', 'epsilon', 'representation_remainder', 'softmax_remainder']
            values = row_arrays(rows, fields)
            for key, value in values.items():
                fact(name+'/finite_'+key, np.isfinite(value).all())
            algebra(name+'/trace', values)
            for key in ['work', 'cost', 'quadratic', 'parameter_work', 'epsilon', 'representation_remainder', 'softmax_remainder']:
                check(name+'/zero_t0_'+key, values[key][0], 0., 0)
            check(name+'/identity_report', [row['identity_maxerror'] for row in rows], np.zeros(plan['updates']))
            check(name+'/summary_identity', summary['identity_maxerror'], 0.)
            for key in FIELDS:
                check(name+'/document_shape_'+key, docs[key].shape, (plan['updates'], plan['batch_size']), 0)
                check(name+'/document_mean_'+key, docs[key].mean(-1), values[key])
            algebra(name+'/documents', docs, False)
            for key in ['work', 'cost', 'quadratic']:
                check(name+'/document_zero_t0_'+key, docs[key][0], np.zeros(plan['batch_size']), 0)
            probes = row_arrays(fixed, fields)
            algebra(name+'/fixed_probe', probes)
            check(name+'/fixed_D_constant', probes['difficulty'], np.repeat(probes['difficulty'][0], len(fixed)), 0)
            for key in ['work', 'cost', 'quadratic', 'parameter_work', 'epsilon']:
                check(name+'/fixed_zero_t0_'+key, probes[key][0], 0., 0)
            token_shape = (plan['measurement']['fixed_probe_documents'], plan['measurement']['target_tokens'])
            for key in FIELDS+['parameter_work', 'epsilon', 'representation_remainder']:
                check(name+'/token_shape_'+key, tokens[key].shape, token_shape, 0)
                check(name+'/token_mean_'+key, tokens[key].mean(), probes[key][-1], 2e-9 if key in TERMS+['representation_remainder'] else 2e-10)
                fact(name+'/token_finite_'+key, np.isfinite(tokens[key]).all())
            algebra(name+'/final_tokens', tokens)
            check(name+'/reported_JVP_gradient_parity', summary['token_A_gradient_dot_maxerror'], 0., 2e-9)
            for key in ['before', 'after']:
                check(name+'/fixture_shape_'+key, fixture[key].shape, (8, plan['measurement']['vocabulary']), 0)
            check(name+'/fixture_target_shape', fixture['target'].shape, (8,), 0)
            if raw_tokens is not None:
                check(name+'/fixture_targets', fixture['target'], raw_tokens[4160, 33:41], 0)
            independent = fixture_terms(fixture['before'], fixture['after'], fixture['target'])
            algebra(name+'/independent_fixture', independent, False)
            for key in FIELDS:
                check(name+'/fixture_vs_final_tokens_'+key, independent[key], tokens[key][0, :8])
            traces[name], token_arrays[name], probe_arrays[name], summaries[name] = values, tokens, probes, summary
            document_arrays[name] = docs
            runs[name] = dict(replica=replica, method=method, windows=window_stats(values, plan),
                              fixed_probe={key: [float(v) for v in value] for key, value in probes.items()},
                              fixed_steps=plan['readouts'], seconds=summary['seconds'],
                              final_response=response_stats({**tokens, 'softmax_remainder':tokens['cost']-tokens['quadratic']}, plan['gates']['response_relative_rms']))
        except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
            failures[name+'/artifact_read_or_structure'] = str(exc)

    pairs = {}
    for replica in range(len(plan['permutation_seeds'])):
        for left, right in itertools.combinations(plan['methods'], 2):
            first, second = f'r{replica}-{left}', f'r{replica}-{right}'
            if first not in traces or second not in traces:
                continue
            label = f'r{replica}/{left}-{right}'
            for scope, arrays in [('trace', traces), ('documents', document_arrays), ('fixed', probe_arrays), ('token', token_arrays)]:
                check(label+'/'+scope+'_shared_D', arrays[first]['difficulty'], arrays[second]['difficulty'], 0)
            difference = {key: traces[first][key]-traces[second][key] for key in traces[first]}
            # Shared D cancels exactly; difference statistics use this zero reference.
            pairs[label] = dict(replica=replica, left=left, right=right,
                                windows=window_stats(difference, plan),
                                final_token_gap=gap_stats(token_arrays[first]['loss']-token_arrays[second]['loss']),
                                final_token_response=response_stats({
                                    key: token_arrays[first][key]-token_arrays[second][key]
                                    for key in token_arrays[first]} | {
                                    'softmax_remainder': (token_arrays[first]['cost']-token_arrays[first]['quadratic'])-
                                    (token_arrays[second]['cost']-token_arrays[second]['quadratic'])}, plan['gates']['paired_relative_rms']))
    table = {}
    for method in plan['methods']:
        selected = [runs[f'r{replica}-{method}']['windows']['full']['raw']
                    for replica in range(len(plan['permutation_seeds'])) if f'r{replica}-{method}' in runs]
        table[method] = {}
        for key in ['difficulty_std', 'parameter_work_mean', 'quadratic_mean', 'epsilon_rms']:
            observations = [value[key] for value in selected]
            table[method][key] = dict(per_replica=observations, minimum=min(observations) if observations else None,
                                      maximum=max(observations) if observations else None)
    expected = len(plan['methods'])*len(plan['permutation_seeds'])
    complete = len(runs) == expected and not incomplete
    report = dict(completed=complete, verification_passed=complete and not failures,
                  available_runs_passed=not failures, expected_runs=expected, completed_runs=len(runs),
                  checks=len(errors), max_error=max(errors.values()) if errors else None,
                  failures=failures, incomplete=incomplete, errors=errors, inputs=manifest,
                  checker_sha256=sha(Path(__file__)),
                  scope='Independent NumPy full-vocabulary final8-token logit reconstruction and all recorded identities, '
                        'paired common D, schedules, hashes and JVP mean parity. No independent training replay or '
                        'gradient/JVP recomputation; parameter trajectories require a separate weight audit.')
    stats = dict(completed=complete, evidence_verified=report['verification_passed'],
                 completed_runs=len(runs), expected_runs=expected, plan_sha256=plan_sha,
                 units='nats; batch8 means over223 targets/document', runs=runs, pairs=pairs, core_table=table,
                 recorded_compute_seconds=sum(value['seconds'] for value in summaries.values()),
                 definitions=dict(blockmean64='Nonoverlapping64-step means; each128-step near/far window has2 block values.',
                                  highpass64='Raw observations minus their own nonoverlapping64-step block mean.',
                                  relative_rms='RMS(epsilon)/RMS(L-D); null if denominator<=1e-14.',
                                  covariance_shares='Cov(term,L-D)/Var(L-D); signed descriptive shares, not causal percentages.',
                                  constant_gap='std/abs(mean)<=.1; near-zero means<=1e-8 marked undefined.',
                                  difficulty_prediction='1-SSE(L-D)/SST(L); pass>=.90; pool is reused.'))
    for filename, value in [('verification.json', report), ('stats.json', stats)]:
        (root/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.data_root)
    print(json.dumps({key: result[key] for key in ['completed', 'verification_passed', 'available_runs_passed',
                     'expected_runs', 'completed_runs', 'checks', 'max_error', 'failures', 'incomplete']}, indent=2))
    raise SystemExit(0 if result['verification_passed'] else 1 if result['failures'] else 2)
