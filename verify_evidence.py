"""Verify public artifact integrity and recorded arithmetic, without training."""
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    webpage = (ROOT / 'docs/index.html').read_text()
    labels = {
        'D_std': ('D_std_nats', 4, False),
        'A_mean': ('A_mean_nats', 6, True),
        'Q2_mean': ('Q2_mean_nats', 6, True),
        'epsilon_rms': ('epsilon_rms_nats', 3, False),
    }
    for algorithm in ['sgd', 'adam']:
        selected = [record for record in visual['records'] if record['algorithm'] == algorithm]
        for label, (metric, digits, signed) in labels.items():
            interval = [min(record[metric] for record in selected), max(record[metric] for record in selected)]
            require(interval == visual['ranges'][algorithm][metric], algorithm + ' visual range ' + metric)
            if label == 'epsilon_rms':
                exponent = -4 if algorithm == 'sgd' else -5
                suffix = ' × 10⁻⁴' if algorithm == 'sgd' else ' × 10⁻⁵'
                expected = '–'.join(f'{value / 10**exponent:.3f}' for value in interval) + suffix
            else:
                formatter = '{:+.' + str(digits) + 'f}' if signed else '{:.' + str(digits) + 'f}'
                separator = ' to ' if signed else '–'
                expected = separator.join(formatter.format(value).replace('-', '−') for value in interval)
            match = re.search(r'data-stat="' + re.escape(algorithm + '.' + label) + r'">([^<]+)<', webpage)
            require(match is not None and match.group(1) == expected, algorithm + ' webpage display ' + label)
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
    print(f'PASS: {checks} release integrity and recorded-arithmetic checks.')
    print('Scope: no gradients, logits, optimizer trajectory, or training were independently recomputed.')


if __name__ == '__main__':
    main()
