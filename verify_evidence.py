"""Verify public artifact integrity and recorded arithmetic, without training."""
import hashlib
import json
import math
import re
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
