"""Verify public artifact integrity and recorded arithmetic, without training."""
import ast
import hashlib
import itertools
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(path.read_text())


def mean(values):
    return math.fsum(values) / len(values)


def rms(values):
    return math.sqrt(mean([value * value for value in values]))


def std(values):
    average = mean(values)
    return rms([value - average for value in values])


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 1e-14 else None


def read_npz(path, require):
    """Read the shipped numeric NPY members without a NumPy dependency."""
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            require(data[:6] == b'\x93NUMPY', str(path) + ' numeric NPY member')
            version = data[6:8]
            require(version in [b'\x01\x00', b'\x02\x00'], str(path) + ' supported NPY version')
            width = 2 if version == b'\x01\x00' else 4
            length = int.from_bytes(data[8:8 + width], 'little')
            offset = 8 + width + length
            header = ast.literal_eval(data[8 + width:offset].decode('latin1'))
            shape = header['shape']
            require(header['descr'] == '<f8' and not header['fortran_order'], str(path) + ' float64 C-order')
            require(len(shape) == 2 and all(isinstance(n, int) and n > 0 for n in shape), str(path) + ' 2D shape')
            require(len(data) - offset == math.prod(shape) * 8, str(path) + ' payload length')
            values = [row[0] for row in struct.iter_unpack('<d', data[offset:])]
            require(all(math.isfinite(value) for value in values), str(path) + ' finite values')
            arrays[name.removesuffix('.npy')] = (shape, values)
    return arrays


def response_stats(values, gate):
    response = [loss - difficulty for loss, difficulty in zip(values['loss'], values['difficulty'])]
    a, q2, eps = [values[key] for key in ['parameter_work', 'quadratic', 'epsilon']]
    response_rms, response_std = rms(response), std(response)
    relative = ratio(rms(eps), response_rms)
    variance = response_std ** 2
    average = mean(response)
    shares, dropped = {}, {}
    for key, term in [('parameter_work', a), ('quadratic', q2), ('epsilon', eps)]:
        term_mean = mean(term)
        covariance = mean([(x - term_mean) * (y - average) for x, y in zip(term, response)])
        shares[key] = ratio(covariance, variance)
        error = rms([m - (x + y + z - d) for m, x, y, z, d in zip(response, a, q2, eps, term)])
        dropped[key] = {'absolute_rms': error, 'relative_rms': ratio(error, response_rms)}
    loss_std = std(values['loss'])
    return {
        'samples': len(response), 'loss_mean': mean(values['loss']), 'loss_std': loss_std,
        'difficulty_mean': mean(values['difficulty']), 'difficulty_std': std(values['difficulty']),
        'response_mean': average, 'response_std': response_std, 'response_rms': response_rms,
        'parameter_work_mean': mean(a), 'quadratic_mean': mean(q2), 'epsilon_rms': rms(eps),
        'response_relative_rms': relative, 'response_gate_passed': relative <= gate if relative is not None else None,
        'difficulty_prediction_fraction': 1 - response_rms ** 2 / loss_std ** 2 if loss_std ** 2 * len(response) > 1e-28 else None,
        'signed_covariance_shares': shares, 'drop_one': dropped,
        'representation_remainder_rms': rms(values['representation_remainder']),
        'softmax_remainder_rms': rms(values['softmax_remainder']),
        'quadratic_vs_KL_relative_rms': ratio(rms([c - q for c, q in zip(values['cost'], q2)]), rms(values['cost'])),
    }


def window_stats(values, plan):
    result = {}
    for label, lo, hi in [('full', 0, plan['updates']), ('near', 0, 128), ('far', plan['updates'] - 128, plan['updates'])]:
        raw = {key: value[lo:hi] for key, value in values.items()}
        blocks = {key: [mean(value[start:start + 64]) for start in range(0, len(value), 64)]
                  for key, value in raw.items()}
        highpass = {key: [value - blocks[key][index // 64] for index, value in enumerate(values)]
                    for key, values in raw.items()}
        result[label] = {mode: response_stats(data, plan['gates']['response_relative_rms'])
                         for mode, data in [('raw', raw), ('highpass64', highpass), ('blockmean64', blocks)]}
        fraction = result[label]['raw']['difficulty_prediction_fraction']
        result[label]['raw']['difficulty_gate_passed'] = fraction >= plan['gates']['difficulty_prediction_fraction'] if fraction is not None else None
    return result


def verify_common(require):
    provenance = load_json(ROOT / 'evidence/common-source-provenance.json')
    webpage = (ROOT / 'docs/index.html').read_text()
    match = re.search(r'<script type="application/json" id="evidence-provenance">(.*?)</script>', webpage, re.S)
    require(match is not None, 'common homepage provenance')
    webpage_sources = json.loads(match.group(1))
    source_records = {(item['protocol'], item['artifact']): item for item in provenance['records']}
    fields = ['loss', 'difficulty', 'work', 'cost', 'quadratic', 'parameter_work', 'epsilon',
              'representation_remainder', 'softmax_remainder']
    completed, saved_checks = 0, 0
    initial_checkpoint = None

    def compare(actual, expected, label):
        if isinstance(actual, dict):
            require(isinstance(expected, dict), label + ' dictionary')
            for key, value in actual.items():
                require(key in expected, label + '/' + key + ' present')
                compare(value, expected[key], label + '/' + key)
        elif isinstance(actual, list):
            require(isinstance(expected, list) and len(actual) == len(expected), label + ' length')
            for index, (left, right) in enumerate(zip(actual, expected)):
                compare(left, right, label + '/' + str(index))
        elif actual is None or isinstance(actual, bool):
            require(actual is expected, label + ' verdict')
        else:
            require(isinstance(expected, (float, int)) and math.isfinite(expected)
                    and math.isclose(actual, expected, rel_tol=2e-9, abs_tol=5e-11), label + ' value')

    def algebra(values, label, has_a=True):
        checks = {
            'W+Q': [l - d - w - q for l, d, w, q in zip(values['loss'], values['difficulty'], values['work'], values['cost'])],
        }
        if has_a:
            checks['R'] = [r - (w - a) for r, w, a in zip(values['representation_remainder'], values['work'], values['parameter_work'])]
            checks['epsilon'] = [e - (l - d - a - q) for e, l, d, a, q in zip(values['epsilon'], values['loss'], values['difficulty'], values['parameter_work'], values['quadratic'])]
            checks['epsilon=R+softmax'] = [e - r - (c - q) for e, r, c, q in zip(values['epsilon'], values['representation_remainder'], values['cost'], values['quadratic'])]
            if 'softmax_remainder' in values:
                checks['softmax'] = [s - (c - q) for s, c, q in zip(values['softmax_remainder'], values['cost'], values['quadratic'])]
        for key, errors in checks.items():
            require(max(map(abs, errors)) < 2e-10, label + '/' + key)
        for key in ['cost', 'quadratic']:
            require(min(values[key]) >= -2e-12, label + '/' + key + ' nonnegative')

    for protocol in ['initial', 'refinement', 'local']:
        root = ROOT / ('evidence/common-' + protocol)
        plan, calibration, stats, report = [load_json(root / (name + '.json')) for name in ['plan', 'calibration', 'stats', 'verification']]
        originals = {}
        for artifact in ['plan.json', 'calibration.json', 'stats.json', 'verification.json', 'parameter-alignment.json']:
            record = source_records[(protocol, artifact)]
            require(hashlib.sha256((root / artifact).read_bytes()).hexdigest() == record['public_sha256'], protocol + '/' + artifact + ' public digest')
            originals[artifact] = record['original_sha256']
        require(stats['plan_sha256'] == calibration['plan_sha256'] == originals['plan.json'], protocol + ' registered plan provenance')
        require(report['completed'] and report['verification_passed'] and not report['failures'] and not report['incomplete'], protocol + ' saved verification verdict')
        require(report['checks'] == len(report['errors']) == 2278 and report['max_error'] == max(report['errors'].values()) < 2e-10
                and report['expected_runs'] == report['completed_runs'] == 18, protocol + ' saved check count')
        saved_checks += report['checks']
        require(stats['completed'] and stats['evidence_verified'] and stats['expected_runs'] == stats['completed_runs'] == 18, protocol + ' statistics completion')
        require(plan['updates'] == 512 and plan['warmup'] == 32 and plan['batch_size'] == 8 and len(plan['permutation_seeds']) == 3 and len(plan['methods']) == 6, protocol + ' registered design')
        require(plan['checkpoint']['step'] == 143000 and plan['dtype'] == 'float64', protocol + ' common late checkpoint')
        if initial_checkpoint is None:
            initial_checkpoint = plan['checkpoint']
        require(all(plan['checkpoint'][key] == initial_checkpoint[key] for key in
                    ['model', 'step', 'revision', 'sha256', 'source_safetensors_sha256', 'content_sha256']), protocol + ' shared checkpoint tensor content')
        require(hashlib.sha256((ROOT / 'experiments/common_checkpoint/worker.py').read_bytes()).hexdigest() == plan['producer_sha256'], protocol + ' producer source')
        require(hashlib.sha256((ROOT / 'experiments/round31_prospective_20261005/worker.py').read_bytes()).hexdigest() == plan['forward_sha256'], protocol + ' forward source')
        compare(calibration['target_Q2'], calibration['rows'][0]['quadratic'], protocol + ' calibration target')
        require(abs(calibration['rows'][-1]['quadratic'] / calibration['target_Q2'] - 1) < .03, protocol + ' calibration match')
        traces, probes, documents, tokens, permutations = {}, {}, {}, {}, {}
        for replica, method in itertools.product(range(3), plan['methods']):
            name = f'r{replica}-{method}'
            directory = root / name
            summary, registration, rows, fixed = [load_json(directory / filename) for filename in ['summary.json', 'registration.json', 'trace.json', 'fixed-probe.json']]
            require(summary['completed'] and summary['method'] == registration['method'] == method and summary['replica'] == registration['replica'] == replica, protocol + '/' + name + ' metadata')
            require(summary['updates'] == registration['updates'] == plan['updates'] and registration['warmup'] == plan['warmup'] and registration['dtype'] == 'float64', name + ' execution contract')
            require(summary['plan_sha256'] == registration['plan_sha256'] == originals['plan.json'] and registration['calibration_sha256'] == originals['calibration.json'], name + ' registration provenance')
            require(summary['producer_sha256'] == plan['producer_sha256'], name + ' producer provenance')
            require(registration['seed'] == plan['permutation_seeds'][replica] and sorted(registration['permutation']) == list(range(4096)), name + ' registered permutation')
            if replica in permutations:
                require(registration['permutation'] == permutations[replica], name + ' paired data order')
            permutations[replica] = registration['permutation']
            require(len(rows) == 512 and [row['step'] for row in rows] == list(range(512)), name + ' update schedule')
            require([row['step'] for row in fixed] == plan['readouts'], name + ' probe schedule')
            rate = calibration['adam_rate'] if method == 'adam_low' else plan['rates'][method]
            compare(registration['rate'], rate, name + ' registered rate')
            scale = .1 if method.startswith('momentum') else 1.
            require(max(abs(row['next_raw_lr'] - rate * min((index + 1) / 32, 1.) * scale) for index, row in enumerate(rows)) < 1e-18, name + ' warmup rates')
            values = {key: [row[key] for row in rows] for key in fields}
            fixed_values = {key: [row[key] for row in fixed] for key in fields}
            require(all(math.isfinite(value) for data in [values, fixed_values] for items in data.values() for value in items), name + ' finite traces')
            for scope, data in [('trace', values), ('probe', fixed_values)]:
                algebra(data, name + '/' + scope)
                require(all(data[key][0] == 0 for key in fields if key not in ['loss', 'difficulty']), name + '/' + scope + ' zero initial response')
            require(len(set(fixed_values['difficulty'])) == 1, name + ' fixed difficulty constant')
            doc_arrays = read_npz(directory / 'document.npz', require)
            token_arrays = read_npz(directory / 'final-tokens.npz', require)
            require(set(doc_arrays) == set(fields[:5]) and set(token_arrays) == set(fields[:-1]), name + ' NPZ fields')
            docs = {key: array for key, (shape, array) in doc_arrays.items()}
            token = {key: array for key, (shape, array) in token_arrays.items()}
            for key, (shape, array) in doc_arrays.items():
                require(shape == (512, 8), name + ' document shape')
                require(max(abs(mean(array[step * 8:(step + 1) * 8]) - values[key][step]) for step in range(512)) < 2e-10, name + ' document means/' + key)
            for key, (shape, array) in token_arrays.items():
                require(shape == (32, 223), name + ' token shape')
                compare(mean(array), fixed_values[key][-1], name + ' token mean/' + key)
            algebra(docs, name + '/document', False)
            require(all(value == 0 for key in ['work', 'cost', 'quadratic'] for value in docs[key][:8]), name + ' zero document response')
            algebra(token, name + '/token')
            token['softmax_remainder'] = [c - q for c, q in zip(token['cost'], token['quadratic'])]
            compare(window_stats(values, plan), stats['runs'][name]['windows'], name + ' aggregate windows')
            compare(response_stats(token, plan['gates']['response_relative_rms']), stats['runs'][name]['final_response'], name + ' final response')
            require(summary['token_A_gradient_dot_maxerror'] <= 2e-9, name + ' recorded JVP mean parity')
            if method == 'head_low':
                require(max(abs(value) for data in [values, fixed_values, token] for value in data['representation_remainder']) < 2e-10, name + ' affine-head control')
            traces[name], probes[name], documents[name], tokens[name] = values, fixed_values, docs, token
            completed += 1
        for replica in range(3):
            for left, right in itertools.combinations(plan['methods'], 2):
                first, second = f'r{replica}-{left}', f'r{replica}-{right}'
                label = f'r{replica}/{left}-{right}'
                for scope, arrays in [('trace', traces), ('probe', probes), ('document', documents), ('token', tokens)]:
                    require(arrays[first]['difficulty'] == arrays[second]['difficulty'], protocol + '/' + label + '/' + scope + ' shared D')
                difference = {key: [x - y for x, y in zip(traces[first][key], traces[second][key])] for key in fields}
                compare(window_stats(difference, plan), stats['pairs'][label]['windows'], label + ' paired response')
                gap = [x - y for x, y in zip(tokens[first]['loss'], tokens[second]['loss'])]
                average, deviation = mean(gap), std(gap)
                relative = ratio(deviation, abs(average)) if abs(average) > 1e-8 else None
                compare({'mean': average, 'std': deviation, 'rms': rms(gap), 'centered_rms': deviation,
                         'absolute_mean': abs(average), 'mean_near_zero': abs(average) <= 1e-8,
                         'std_over_absolute_mean': relative, 'constant_gap_gate_passed': relative <= .1 if relative is not None else None,
                         'zero_gap': rms(gap) <= 1e-10}, stats['pairs'][label]['final_token_gap'], label + ' token constant-gap test')
                if protocol == 'local':
                    require(relative is not None and relative > .1, label + ' retained terminal constant-gap failure')
        table = load_json(root / 'figures/core-table.json')
        require(not table['partial'] and len(table['records']) == 24, protocol + ' complete core table')
        for record in table['records']:
            key, method = record['metric'], record['method']
            actual = [response_stats(traces[f'r{replica}-{method}'], .1)[key] for replica in range(3)]
            compare({'per_replica': actual, 'minimum': min(actual), 'maximum': max(actual)}, stats['core_table'][method][key], protocol + ' table ranges/' + key + '/' + method)
            compare([row['value'] for row in record['replicas']], actual, protocol + ' figure table replicas')
            compare(record['minimum'], min(actual), protocol + ' figure table minimum')
            compare(record['maximum'], max(actual), protocol + ' figure table maximum')
        for path, digest in webpage_sources[protocol]['sources'].items():
            if path == 'figures/plot-manifest.json':
                require(hashlib.sha256((root / 'plot-manifest.json').read_bytes()).hexdigest() == digest, protocol + ' page figure manifest')
            else:
                require(digest == originals[path], protocol + ' page original source digest/' + path)
        for path, digest in webpage_sources[protocol]['figures'].items():
            require(hashlib.sha256((root / 'figures' / path).read_bytes()).hexdigest() == digest, protocol + ' page figure digest/' + path)
        if protocol == 'local':
            table_match = re.search(r'<table class="numbers">(.*?)</table>', webpage, re.S)
            require(table_match is not None, 'native homepage numerical table')
            table_rows = re.findall(r'<tr>(.*?)</tr>', table_match.group(1), re.S)[1:]
            metrics = ['difficulty_std', 'parameter_work_mean', 'quadratic_mean', 'epsilon_rms']
            require(len(table_rows) == len(metrics), 'homepage table four terms')
            for metric, row in zip(metrics, table_rows):
                cells = re.findall(r'<td>(.*?)</td>', row, re.S)
                require(len(cells) == len(plan['methods']), 'homepage table six settings')
                for method, cell in zip(plan['methods'], cells):
                    cell = re.sub(r'<span class="mobile-setting">.*?</small></span>', '', cell, flags=re.S)
                    low, high = [stats['core_table'][method][metric][key] for key in ['minimum', 'maximum']]
                    if abs(high) < .01 and low != 0:
                        exponent = int(f'{max(abs(low), abs(high)):.0e}'.split('e')[1])
                        low_text = f'{low / 10**exponent:.4g}'.replace('-', '−')
                        high_text = f'{high / 10**exponent:.4g}'.replace('-', '−')
                        interval = f'({low_text} to {high_text}) × 10<sup>{exponent}</sup>'
                    else:
                        interval = f'{low:.4g}–{high:.4g}'.replace('-', '−')
                    require(cell == interval, 'homepage measured ' + metric + '/' + method)
            for method in plan['methods']:
                for replica in range(3):
                    windows = stats['runs'][f'r{replica}-{method}']['windows']
                    require(windows['full']['raw']['response_gate_passed'] == (method in ['adam_low', 'head_low']), method + ' terminal closure verdict')
                    if method in ['sgd_low', 'momentum_low']:
                        require(windows['near']['raw']['response_gate_passed'] and not windows['far']['raw']['response_gate_passed'], method + ' short-window scope retained')
    for asset, protocol, source in [
        ('common-local-arriving.svg', 'local', 'four-terms-arriving.svg'),
        ('common-local-fixed-probe.svg', 'local', 'four-terms-fixed-probe.svg'),
        ('common-initial-arriving.svg', 'initial', 'four-terms-arriving.svg'),
        ('common-refinement-arriving.svg', 'refinement', 'four-terms-arriving.svg'),
    ]:
        require(hashlib.sha256((ROOT / 'docs/assets' / asset).read_bytes()).hexdigest() == webpage_sources[protocol]['figures'][source], 'published homepage chart/' + asset)
    require(completed == provenance['completed_runs'] == 54 and saved_checks == 6834, 'all common-checkpoint runs and recorded checks')
    require(provenance['preserved_failed_protocols'], 'failed adaptive protocols preserved')
    require('Sole core author' in webpage and 'Corresponding author' in webpage and 'table class="definitions"' in webpage, 'native visual abstract and author roles')
    return completed


def main():
    checks = 0

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise SystemExit('FAIL: ' + message)

    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        require(path.is_file(), 'missing ' + item['path'])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'hash ' + item['path'])

    summary = json.loads((ROOT / 'evidence/numerical-theory-summary.json').read_text())
    arrivals = json.loads((ROOT / 'evidence/arrival-analysis.json').read_text())
    require(summary['completed'] is True and arrivals['completed'] is True, 'analysis completion')
    for window in ['near', 'far']:
        records = [row for row in summary['frozen_difficulty'] if row['window'] == window]
        require(len(records) == 6, window + ' frozen rows')
        low, high = min(row['common_reference_explained'] for row in records), max(row['common_reference_explained'] for row in records)
        bounds = {'near': (96.657, 97.578), 'far': (96.745, 97.582)}[window]
        require(round(100 * low, 3) == bounds[0] and round(100 * high, 3) == bounds[1], window + ' common-reference range')

    html = (ROOT / 'docs/reports/rgi-followup-panel.html').read_text()
    match = re.search(r'<script id="experiment-data" type="application/json">(.*?)</script>', html, re.S)
    require(match is not None, 'embedded panel data')
    panel = json.loads(match.group(1))
    require(panel['summary'] == summary, 'panel and supplied summary')
    for key, rows in panel['series'].items():
        require(len(rows) == 512, key + ' trace length')
        for row in rows:
            step, loss, difficulty, shared, work_a, quadratic, work_w, cost, remainder = row
            require(abs((loss - difficulty) - (work_w + cost)) < 5e-11, key + ' logit identity')
            require(abs(remainder - (work_w - work_a)) < 5e-11, key + ' residual identity')
        for window, (lo, hi) in {'near': (0, 128), 'far': (384, 512), 'full': (0, 512)}.items():
            selected = rows[lo:hi]
            actual = [row[1] - row[2] for row in selected]
            prediction = [row[4] + row[5] for row in selected]
            mse = sum((a - b) ** 2 for a, b in zip(actual, prediction)) / len(actual)
            signal = sum(a * a for a in actual) / len(actual)
            relative = math.sqrt(mse / signal)
            ref = next(row for row in summary['slow_response'] if row['run'] == 'arrival-' + key and row['window'] == window)
            require(abs(relative - ref['relative_rms']) < 5e-8, key + ' ' + window + ' RMS')
            require(relative <= .1, key + ' ' + window + ' finite-response gate')
    visual = json.loads((ROOT / 'docs/assets/visual-abstract-data.json').read_text())
    require(visual['table_window'] == [0, 512], 'visual table window')
    require(len(visual['records']) == 6, 'visual table run count')
    for path, digest in visual['sources'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, 'visual source ' + path)
    for record in visual['records']:
        key = record['run'].removeprefix('arrival-')
        rows = panel['series'][key]
        mean = lambda values: sum(values) / len(values)
        difficulty = [row[2] for row in rows]
        difficulty_mean = mean(difficulty)
        measured = {
            'D_std_nats': math.sqrt(mean([(value - difficulty_mean) ** 2 for value in difficulty])),
            'A_mean_nats': mean([row[4] for row in rows]),
            'Q2_mean_nats': mean([row[5] for row in rows]),
            'epsilon_rms_nats': math.sqrt(mean([(row[1] - row[2] - row[4] - row[5]) ** 2 for row in rows])),
        }
        for metric, actual in measured.items():
            require(abs(actual - record[metric]) < 5e-10, key + ' visual ' + metric)
    # These figures describe the historical distinct-start experiment.
    for algorithm in ['sgd', 'adam']:
        selected = [record for record in visual['records'] if record['algorithm'] == algorithm]
        for metric in ['D_std_nats', 'A_mean_nats', 'Q2_mean_nats', 'epsilon_rms_nats']:
            interval = [min(record[metric] for record in selected), max(record[metric] for record in selected)]
            require(interval == visual['ranges'][algorithm][metric], algorithm + ' visual range ' + metric)
    for language in ['en', 'zh']:
        svg = ET.parse(ROOT / ('docs/assets/visual-abstract-' + language + '.svg')).getroot()
        require(svg.find('{http://www.w3.org/2000/svg}title') is not None, language + ' visual SVG title')
    paired = summary['paired_incremental_failure']
    require(len(paired) == 3 and all(row['relative_rms'] > .1 for row in paired), 'optimizer-difference failure retained')
    for record in summary['verification']:
        require(record['passed'] is True, 'saved verifier verdict')
    require(sum(record['checks'] for record in summary['verification']) == 27720, 'saved check count')
    require(arrivals['algorithm_aggregate']['sgd']['far']['highpass64']['comparisons']['A_plus_Q2']['ten_percent_rms_pass_count'] == 0,
            'SGD high-pass failure retained')
    require(arrivals['algorithm_aggregate']['sgd']['far']['blockmean64']['comparisons']['A_plus_Q2']['ninety_percent_pass_count'] == 1,
            'SGD far-block variance limitation retained')
    common_runs = verify_common(require)
    print(f'PASS: {checks} release integrity and recorded-arithmetic checks; {common_runs} common-checkpoint runs.')
    print('Scope: public trace, document, and token arithmetic was independently recomputed. '
          'No gradients, full-vocabulary logits, optimizer trajectory, or training were replayed. '
          'Scientific failures are checked and retained, not treated as integrity failures.')


if __name__ == '__main__':
    main()
