"""Plot verified common-checkpoint response terms and the core numerical table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = {
    'sgd_low': ('SGD · smaller rate', '#0072B2', 'o'),
    'momentum_low': ('Heavy Ball · smaller', '#E69F00', '^'),
    'adam_low': ('Adam · calibrated', '#009E73', 'D'),
    'sgd_high': ('SGD · larger rate', '#56B4E9', 's'),
    'momentum_high': ('Heavy Ball · larger', '#D55E00', 'v'),
    'head_low': ('Head only', '#CC79A7', 'X'),
}
LOW_METHODS = ['sgd_low', 'momentum_low', 'adam_low', 'head_low']
OVERVIEW_STOP = 300
ZOOM_STOP = 310
PRESENTATION_BLOCK_SIZE = 20
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
        'font.family': 'DejaVu Sans', 'font.size': 9.2, 'axes.titlesize': 10,
        'axes.labelsize': 9.2, 'legend.fontsize': 9, 'axes.spines.top': False,
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
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(destination / f'{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


def blockmeans(x, y):
    if len(y) % 64:
        raise ValueError('Arriving traces must contain whole preregistered 64-step blocks.')
    return x.reshape(-1, 64).mean(-1), y.reshape(-1, 64).mean(-1)


def method_style(method, marker=True):
    _, color, symbol = METHODS[method]
    return dict(color=color, ls='-', lw=1.3,
                marker=symbol if marker else None, ms=4.6, mew=1.0,
                markerfacecolor='none' if method.startswith('momentum') else color,
                markeredgecolor=color)


def validate_shared(traces, name, fixed):
    shared = next(iter(traces.values()))
    for method, values in traces.items():
        if not np.array_equal(shared['step'], values['step']):
            raise ValueError(f'{name}: mismatched measurement steps for {method}.')
        if not np.allclose(shared['difficulty'], values['difficulty'], rtol=0, atol=2e-10):
            raise ValueError(f'{name}: frozen difficulty is not shared for {method}.')
        for field, _, _ in PANELS[1:]:
            if values[field][0] != 0 or values['step'][0] != 0:
                raise ValueError(f'{name}: {method}/{field} must be measured from zero.')
    if fixed and not np.allclose(shared['difficulty'], shared['difficulty'][0], rtol=0, atol=2e-10):
        raise ValueError('Fixed-probe frozen difficulty must be constant.')
    return shared


def arriving_panels(traces, plan, calibration, destination, name, status):
    shared = validate_shared(traces, name, fixed=False)
    x = shared['step']
    overview = (x >= 0) & (x <= OVERVIEW_STOP)
    coarse = (x >= 0) & (x < OVERVIEW_STOP)
    zoom = (x >= OVERVIEW_STOP) & (x <= ZOOM_STOP)
    if not np.array_equal(x[coarse], np.arange(OVERVIEW_STOP)):
        raise ValueError('Coarse presentation requires observed steps 0–299.')
    if not np.array_equal(x[zoom], np.arange(OVERVIEW_STOP, ZOOM_STOP+1)):
        raise ValueError('Zoom requires all eleven observed steps 300–310.')
    coarse_x = x[coarse].reshape(-1, PRESENTATION_BLOCK_SIZE).mean(-1)
    fig, axes = plt.subplots(4, 2, figsize=(6.8, 7.85), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(left=.10, right=.978, bottom=.065, top=.835, hspace=.59, wspace=.39)
    handles = []
    for row, (field, title, ylabel) in enumerate(PANELS):
        left, right = axes[row]
        short_title = title[4:]
        left.set(title=title, ylabel=ylabel, xlim=(0, OVERVIEW_STOP),
                 xticks=[0, 60, 120, 180, 240, 300])
        right.set(title=short_title, ylabel=ylabel, xlim=(OVERVIEW_STOP, ZOOM_STOP),
                  xticks=np.arange(OVERVIEW_STOP, ZOOM_STOP+1, 2))
        if row == 0:
            left.set_title('0–300: coarse overview\n'+title, pad=8)
            right.set_title('300–310: every step\n'+short_title, pad=8)
        left.axvspan(0, plan['warmup'], color='#777777', alpha=.09, lw=0, zorder=0)
        for ax in [left, right]:
            if row:
                ax.axhline(0, color='#777777', lw=.6, zorder=0)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
            ax.grid(axis='y', alpha=.16, lw=.5)
        if field == 'difficulty':
            left.plot(x[overview], shared[field][overview], color='#222222', alpha=.23, lw=.55)
            coarse_y = shared[field][coarse].reshape(-1, PRESENTATION_BLOCK_SIZE).mean(-1)
            left.plot(coarse_x, coarse_y, color='#222222', marker='o', ms=3.5, lw=1.35)
            right.plot(x[zoom], shared[field][zoom], color='#222222', marker='o', ms=4, lw=1.35)
            continue
        for method, values in traces.items():
            y = values[field]
            style = method_style(method)
            left.plot(x[overview], y[overview], color=style['color'], alpha=.18, lw=.55)
            left.plot([0], [0], **style, zorder=4)
            coarse_y = y[coarse].reshape(-1, PRESENTATION_BLOCK_SIZE).mean(-1)
            line, = left.plot(coarse_x, coarse_y, label=method_label(method, plan, calibration), **style)
            right.plot(x[zoom], y[zoom], **style)
            if field == 'parameter_work':
                handles.append(line)
        if field == 'quadratic' and len(traces) > len(LOW_METHODS):
            low_traces = {method: values for method, values in traces.items() if method in LOW_METHODS}
            low_peak = max(values[field][zoom].max() for values in low_traces.values())
            all_peak = max(values[field][zoom].max() for values in traces.values())
            if all_peak > 5*low_peak:
                inset = right.inset_axes([.23, .18, .70, .34])
                for method, values in low_traces.items():
                    style = method_style(method)
                    style.update(ms=3.1, lw=.9, mew=.7)
                    inset.plot(x[zoom], values[field][zoom], **style)
                inset.set(title='Smaller / calibrated', xlim=(OVERVIEW_STOP, ZOOM_STOP),
                          ylim=(0, 1.15*low_peak), xticks=[])
                inset.title.set_fontsize(9)
                inset.tick_params(labelsize=9, pad=1, length=2)
                inset.yaxis.set_major_locator(MaxNLocator(3))
                inset.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                inset.yaxis.get_offset_text().set_fontsize(9)
                inset.grid(axis='y', alpha=.15, lw=.5)
                inset.patch.set_alpha(.95)
    for ax in axes[-1]:
        ax.set_xlabel('Continuation updates')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.002),
               ncol=3, frameon=False, columnspacing=1.1, handlelength=1.8)
    save(fig, destination, name, status)


def four_panels(traces, plan, calibration, destination, name, status, fixed=False, replica=0, subset=''):
    if not fixed:
        arriving_panels(traces, plan, calibration, destination, name, status)
        return
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.65))
    axes_style(axes, plan)
    shared = validate_shared(traces, name, fixed=True)
    legend_handles = []
    for ax, (field, title, ylabel) in zip(axes.flat, PANELS):
        ax.set(title=title, ylabel=ylabel)
        x = shared['step']
        if field == 'difficulty':
            ax.plot(x, shared[field], color='#222222', marker='o', ms=4, lw=1.7)
            continue
        for method, values in traces.items():
            label = method_label(method, plan, calibration)
            y = values[field]
            line, = ax.plot(x, y, label=label, **method_style(method))
            if field == 'parameter_work':
                legend_handles.append(line)
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=3, frameon=False, columnspacing=1.1, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, .79))
    save(fig, destination, name, status)


def paired_plot(traces, plan, calibration, destination, status):
    pairs = [('sgd_low', 'momentum_low', f'SGD − Heavy Ball\neffective h={plan["rates"]["sgd_low"]:.3g}', '#0072B2'),
             ('sgd_high', 'momentum_high', f'SGD − Heavy Ball\neffective h={plan["rates"]["sgd_high"]:.3g}', '#D55E00'),
             ('adam_low', 'head_low', f'Adam − Head only\nraw η={calibration["adam_rate"]:.3g} / {plan["rates"]["head_low"]:.3g}', '#009E73')]
    pairs = [pair for pair in pairs if pair[0] in traces and pair[1] in traces]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.05))
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
    axes[0].set(title='(a) Measured and reconstructed', ylabel='Difference (nats)')
    axes[1].set(title=r'(b) Paired remaining response $\Delta\varepsilon$', ylabel='Difference (nats)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=3, frameon=False, columnspacing=1.)
    axes[0].legend(handles=[Line2D([], [], color='#222222', lw=1.5, label=r'Measured $\Delta L$'),
                            Line2D([], [], color='#222222', lw=1.5, ls='--', label=r'$\Delta A+\Delta Q_2$')],
                   loc='lower left', frameon=False, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, .79))
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
    low_methods = LOW_METHODS
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
                    batch_block_size=64,
                    smoothing=('No fitted smoothing. Arriving overview: 20-step presentation means; '
                               'paired plot: unchanged registered 64-step means; zoom/fixed probe: raw values.'),
                    presentation=dict(
                        arriving_layout='Four rows × two equal-width columns',
                        overview_raw_steps=[0, OVERVIEW_STOP],
                        overview_mean_steps=[0, OVERVIEW_STOP-1],
                        overview_nonoverlapping_mean_size=PRESENTATION_BLOCK_SIZE,
                        overview_means_are_presentation_only=True,
                        origin='Actual step-zero markers are separate from the first block mean at step 9.5.',
                        zoom_inclusive_steps=[OVERVIEW_STOP, ZOOM_STOP],
                        zoom_observations=ZOOM_STOP-OVERVIEW_STOP+1,
                        setting_lines='Solid; distinct Okabe–Ito colors and marker shapes, no jitter or rescaling.',
                        variance_inset='Smaller/calibrated settings only when the full linear scale compresses them.',
                        paired_lines='Solid measured ΔL; dashed ΔA + ΔQ2; neither encodes a setting rate.',
                        embedded_prose_caption=False),
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
