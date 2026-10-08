"""Plot verified common-checkpoint response terms and the core numerical table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = {
    'sgd_low': ('SGD (smaller rate)', '#0072B2', '-'),
    'momentum_low': ('Heavy Ball (smaller rate)', '#D55E00', '-'),
    'adam_low': ('Adam (calibrated)', '#009E73', '-'),
    'sgd_high': ('SGD (larger rate)', '#0072B2', '--'),
    'momentum_high': ('Heavy Ball (larger rate)', '#D55E00', '--'),
    'head_low': ('Head only', '#CC79A7', '-.'),
}
PANELS = [
    ('difficulty', r'(a) Shared frozen difficulty $D$', r'$D$ (nats)'),
    ('parameter_work', r'(b) Linear response $A$', r'$A$ (nats)'),
    ('quadratic', r'(c) Logit variance $Q_2$', r'$Q_2$ (nats)'),
    ('epsilon', r'(d) Remaining response $\varepsilon$', r'$\varepsilon$ (nats)'),
]
TABLE_ROWS = [
    ('difficulty_std', r'Batch difficulty $D$: std.'),
    ('parameter_work_mean', r'Linear response $A$: mean'),
    ('quadratic_mean', r'Logit variance $Q_2$: mean'),
    ('epsilon_rms', r'Remaining response $\varepsilon$: RMS'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(rows):
    return {key: np.asarray([row[key] for row in rows], dtype=float)
            for key in ['step', 'loss', 'difficulty', 'parameter_work', 'quadratic', 'epsilon']}


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'legend.fontsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'axes.formatter.useoffset': False,
    })


def method_label(method, plan, calibration):
    label = METHODS[method][0]
    rate = calibration['adam_rate'] if method == 'adam_low' else plan['rates'][method]
    symbol = 'raw η' if method in ['adam_low', 'head_low'] else 'effective h'
    return label+'\n'+f'{symbol}={rate:.3g}'


def protocol_label(plan):
    if not plan.get('adaptive_parent_plan_sha256'):
        return 'Initial registered protocol'
    if plan.get('local_scale_contract_sha256'):
        return 'Terminal local-scale protocol (adaptive)'
    return 'Rate-refinement protocol (adaptive)'


def axes_style(axes, plan, terms=True):
    for index, ax in enumerate(np.asarray(axes).flat):
        ax.axvspan(0, plan['warmup'], color='#777777', alpha=.09, lw=0, zorder=0)
        if terms and index:
            ax.axhline(0, color='#777777', lw=.7, zorder=1)
        ax.set_xlim(0, plan['updates'])
        ax.set_xlabel('Continuation updates')
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.grid(axis='y', alpha=.15, lw=.6)


def save(fig, destination, name, status):
    if status:
        fig.suptitle(status, color='#8C4513', fontsize=11, y=1.045)
    fig.tight_layout(rect=(0, .13, 1, .81))
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(destination / f'{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


def blockmeans(x, y):
    if len(y) % 64:
        raise ValueError('Arriving traces must contain whole preregistered 64-step blocks.')
    return x.reshape(-1, 64).mean(-1), y.reshape(-1, 64).mean(-1)


def four_panels(traces, plan, calibration, destination, name, status, fixed=False, replica=0, subset=''):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
    axes_style(axes, plan)
    shared = next(iter(traces.values()))
    for key, values in traces.items():
        if not np.array_equal(shared['step'], values['step']):
            raise ValueError(f'{name}: mismatched measurement steps for {key}.')
        if not np.allclose(shared['difficulty'], values['difficulty'], rtol=0, atol=2e-10):
            raise ValueError(f'{name}: frozen difficulty is not shared for {key}.')
    if fixed and not np.allclose(shared['difficulty'], shared['difficulty'][0], rtol=0, atol=2e-10):
        raise ValueError('Fixed-probe frozen difficulty must be constant.')
    legend_handles = []
    for ax, (field, title, ylabel) in zip(axes.flat, PANELS):
        ax.set(title=title, ylabel=ylabel)
        x = shared['step']
        if field == 'difficulty':
            if fixed:
                ax.plot(x, shared[field], color='#222222', marker='o', ms=4, lw=1.7)
            else:
                ax.plot(x, shared[field], color='#222222', alpha=.22, lw=.65)
                bx, by = blockmeans(x, shared[field])
                ax.plot(bx, by, color='#222222', marker='o', ms=4, lw=1.7)
            continue
        for method, values in traces.items():
            _, color, linestyle = METHODS[method]
            label = method_label(method, plan, calibration)
            y = values[field]
            if y[0] != 0 or values['step'][0] != 0:
                raise ValueError(f'{name}: {method}/{field} must be measured from zero.')
            if fixed:
                line, = ax.plot(x, y, color=color, ls=linestyle, marker='o', ms=3.8,
                                lw=1.65, label=label)
            else:
                ax.plot(x, y, color=color, ls=linestyle, alpha=.22, lw=.65)
                ax.plot(x[0], y[0], color=color, marker='o', ms=3.5)
                bx, by = blockmeans(x, y)
                line, = ax.plot(bx, by, color=color, ls=linestyle, marker='o', ms=3.5,
                                lw=1.65, label=label)
            if field == 'parameter_work':
                legend_handles.append(line)
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=3, frameon=False, columnspacing=1.8)
    caption = ('Dots: fixed held-out 32-document readouts; frozen D is horizontal.' if fixed else
               'Light lines: all observed arriving batches. Solid/dashed lines with dots: nonoverlapping 64-step means.')
    scope = protocol_label(plan)+f'; permutation {replica+1} of {len(plan["permutation_seeds"])}.'
    if subset:
        scope += ' '+subset
    fig.text(.5, .025, caption+'\n'+scope+'\n'
             'Heavy Ball raw η = 0.1h; Adam rate matches one reset-step Q₂, not its whole trajectory.\n'
             'Grey: 32-update warmup. Shared batch order; separate vertical scales; all nats. Rate labels do not imply stability regimes.',
             ha='center', va='center', fontsize=8.5)
    save(fig, destination, name, status)


def paired_plot(traces, plan, calibration, destination, status):
    pairs = [('sgd_low', 'momentum_low', f'SGD − Heavy Ball\neffective h={plan["rates"]["sgd_low"]:.3g}', '#0072B2'),
             ('sgd_high', 'momentum_high', f'SGD − Heavy Ball\neffective h={plan["rates"]["sgd_high"]:.3g}', '#D55E00'),
             ('adam_low', 'head_low', f'Adam − Head only (different settings)\nraw η={calibration["adam_rate"]:.3g} / {plan["rates"]["head_low"]:.3g}', '#009E73')]
    pairs = [pair for pair in pairs if pair[0] in traces and pair[1] in traces]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))
    axes_style(axes, plan, terms=False)
    for left, right, label, color in pairs:
        a, b = traces[left], traces[right]
        x = a['step']
        loss = a['loss']-b['loss']
        response = a['parameter_work']-b['parameter_work']+a['quadratic']-b['quadratic']
        remainder = a['epsilon']-b['epsilon']
        if not np.allclose(loss-response, remainder, rtol=0, atol=2e-10):
            raise ValueError(f'Paired response does not close for {left}/{right}.')
        bx, loss_mean = blockmeans(x, loss)
        _, response_mean = blockmeans(x, response)
        axes[0].plot(x, loss, color=color, alpha=.2, lw=.65)
        axes[0].plot(bx, loss_mean, color=color, marker='o', ms=3.5, lw=1.65, label=label)
        axes[0].plot(bx, response_mean, color=color, ls='--', lw=1.5)
        axes[1].plot(x, remainder, color=color, alpha=.2, lw=.65)
        _, remainder_mean = blockmeans(x, remainder)
        axes[1].plot(bx, remainder_mean, color=color, marker='o', ms=3.5, lw=1.65)
    for ax in axes:
        ax.axhline(0, color='#777777', lw=.7)
    axes[0].set(title='(a) Paired measured response and reconstruction', ylabel='Difference (nats)')
    axes[1].set(title=r'(b) Paired remaining response $\Delta\varepsilon$', ylabel='Difference (nats)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=3, frameon=False, columnspacing=1.)
    fig.text(.5, .025, 'Light lines: observed values. Dots: 64-step means. Panel (a): solid ΔL; dashed ΔA + ΔQ₂.\n'
             'Common difficulty cancels. Original shared step axis; no fitted time alignment.\n'+protocol_label(plan)+'.',
             ha='center', va='center', fontsize=8.5)
    save(fig, destination, 'paired-response', status)


def core_table(stats, plan, calibration, destination, partial):
    rows, records = [], []
    replica_count = len(plan['permutation_seeds'])
    for metric, label in TABLE_ROWS:
        cells = []
        for method in plan['methods']:
            entry = stats['core_table'][method][metric]
            replicas = []
            for replica in range(replica_count):
                run = stats['runs'].get(f'r{replica}-{method}')
                if run is not None:
                    value = run['windows']['full']['raw'][metric]
                    replicas.append({'replica': replica, 'value': value})
            recorded = [item['value'] for item in replicas]
            if recorded != entry['per_replica']:
                raise ValueError(f'Core table replica records disagree for {method}/{metric}.')
            if recorded:
                low, high = min(recorded), max(recorded)
                cell = f'{low:.5g}--{high:.5g}' if len(recorded) > 1 else f'{low:.5g}'
            else:
                low, high, cell = None, None, r'\textemdash'
            if partial:
                cell += r'\,{' + f'({len(recorded)}/{replica_count})' + '}'
            cells.append(cell)
            records.append({'metric': metric, 'method': method, 'replicas': replicas,
                            'minimum': low, 'maximum': high, 'completed_replicas': len(recorded),
                            'expected_replicas': replica_count,
                            'registered_rate': calibration['adam_rate'] if method == 'adam_low' else plan['rates'][method],
                            'rate_kind': 'raw' if method in ['adam_low', 'head_low'] else 'effective h'})
        rows.append(label+' & '+' & '.join(cells)+r' \\')
    names = {'sgd_low': r'SGD$_{\rm low}$', 'momentum_low': r'HB$_{\rm low}$',
             'adam_low': 'Adam', 'sgd_high': r'SGD$_{\rm high}$',
             'momentum_high': r'HB$_{\rm high}$', 'head_low': 'Head only'}
    tex = '\n'.join([
        '% Generated from independently verified batch records; every value is in nats.',
        r'\begin{tabular}{l'+'r'*len(plan['methods'])+'}', r'\hline',
        'Term and statistic & '+' & '.join(names[method] for method in plan['methods'])+r' \\',
        r'\hline', *rows, r'\hline', r'\end{tabular}',
        '% Ranges span the three paired permutations of one reused data pool; they are not confidence intervals.',
        '% '+protocol_label(plan)+'. Method low/high names are relative rates, not established stability regimes.',
        '% Rates: '+', '.join(method_label(method, plan, calibration).replace('\n', ': ') for method in plan['methods']),
        '% Partial table: parentheses give observed/expected permutations.' if partial else '',
    ])+'\n'
    (destination/'core-table.tex').write_text(tex)
    artifact = dict(units='nats', partial=partial, records=records, protocol=protocol_label(plan),
                    scope='Full 512-update arriving-batch means; ranges across paired permutations of one reused pool.')
    (destination/'core-table.json').write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    with (destination/'core-table.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['metric', 'method', 'replica', 'value', 'units'])
        writer.writeheader()
        for record in records:
            for value in record['replicas']:
                writer.writerow(dict(metric=record['metric'], method=record['method'],
                                     replica=value['replica'], value=value['value'], units='nats'))


def main(args):
    root = args.root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    verification_bytes = (root/'verification.json').read_bytes()
    stats_bytes = (root/'stats.json').read_bytes()
    verification = json.loads(verification_bytes)
    stats = json.loads(stats_bytes)
    if not verification['available_runs_passed'] or verification['failures']:
        raise ValueError('Independent checker has unresolved evidence failures.')
    if not args.partial and not (verification['verification_passed'] and stats['evidence_verified']):
        raise ValueError('All registered runs must pass verification; use --partial only for a labeled preview.')
    plan_sha = sha(root/'plan.json')
    if stats['plan_sha256'] != plan_sha:
        raise ValueError('Statistics refer to a different plan.')
    if stats['completed_runs'] != verification['completed_runs']:
        raise ValueError('Statistics and verification have different completed-run counts; rerun the checker.')
    verified_inputs = {item['path']: item['sha256'] for item in verification['inputs']}
    inputs = []

    def read(relative):
        path = root/relative
        digest = sha(path)
        if verified_inputs.get(relative) != digest:
            raise ValueError(f'File changed since verification: {relative}.')
        inputs.append(dict(path=relative, sha256=digest))
        return json.loads(path.read_text())

    read('plan.json')
    calibration = read('calibration.json')
    completed = {replica: [method for method in plan['methods'] if f'r{replica}-{method}' in stats['runs']]
                 for replica in range(len(plan['permutation_seeds']))}
    available = [replica for replica, methods in completed.items() if methods]
    if not available:
        raise ValueError('No complete independently checked science runs; no plot or table was created.')
    replica = args.replica if args.replica is not None else 0
    if replica not in available:
        raise ValueError(f'Replica {replica} has no complete checked runs.')
    traces, probes = {}, {}
    for method in completed[replica]:
        run = f'r{replica}-{method}'
        traces[method] = arrays(read(run+'/trace.json'))
        probes[method] = arrays(read(run+'/fixed-probe.json'))
    partial = not stats['completed'] or not stats['evidence_verified']
    status = (f'PARTIAL EVIDENCE — {stats["completed_runs"]}/{stats["expected_runs"]} runs; '
              f'plotted permutation {replica+1}, {len(traces)}/{len(plan["methods"])} settings') if partial else ''
    configure()
    destination = root/'figures'
    destination.mkdir(exist_ok=True)
    four_panels(traces, plan, calibration, destination, 'four-terms-arriving', status, replica=replica)
    four_panels(probes, plan, calibration, destination, 'four-terms-fixed-probe', status, fixed=True, replica=replica)
    low_methods = ['sgd_low', 'momentum_low', 'adam_low', 'head_low']
    low_traces = {method: values for method, values in traces.items() if method in low_methods}
    low_probes = {method: values for method, values in probes.items() if method in low_methods}
    if len(low_traces) > 1:
        subset = 'Smaller-rate/calibrated settings only; larger-rate results are in the full figure.'
        low_status = (f'PARTIAL EVIDENCE — {stats["completed_runs"]}/{stats["expected_runs"]} runs; '
                      f'plotted permutation {replica+1}, {len(low_traces)}/4 low/calibrated settings') if partial else ''
        four_panels(low_traces, plan, calibration, destination, 'four-terms-arriving-low-settings', low_status,
                    replica=replica, subset=subset)
        four_panels(low_probes, plan, calibration, destination, 'four-terms-fixed-probe-low-settings', low_status,
                    fixed=True, replica=replica, subset=subset)
    paired_plot(traces, plan, calibration, destination, status)
    core_table(stats, plan, calibration, destination, partial)
    source = Path(__file__).resolve()
    manifest = dict(source_sha256=sha(source), plan_sha256=plan_sha, inputs=inputs,
                    stats_sha256=hashlib.sha256(stats_bytes).hexdigest(),
                    verification_sha256=hashlib.sha256(verification_bytes).hexdigest(),
                    partial=partial, plotted_replica=replica, methods=list(traces),
                    protocol=protocol_label(plan), rates=plan['rates'], adam_raw_rate=calibration['adam_rate'],
                    batch_block_size=64, smoothing='None; raw traces and prespecified nonoverlapping block means.',
                    figures=[dict(path=path.name, sha256=sha(path)) for path in sorted(destination.iterdir())
                             if path.suffix in ['.pdf', '.svg', '.png', '.tex', '.csv', '.json']
                             and path.name != 'plot-manifest.json'],
                    scope='Observed realized response, with independent vertical scales; no fitted time warp.')
    (destination/'plot-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(figures=str(destination), partial=partial, replica=replica, methods=list(traces))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--replica', type=int)
    main(parser.parse_args())
