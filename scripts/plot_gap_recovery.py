#!/usr/bin/env python3
"""Plot structural development coverage; no model output or source-value access."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--route-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    assert report['schema'] == 'hydra-development-gap-recovery/v1'
    route_raw = args.route_report.read_bytes()
    route = json.loads(route_raw)
    assert route['schema'] == 'hydra-development-gap-routing/v1'
    assert route['parent_census_sha256'] == hashlib.sha256(raw).hexdigest()
    names = {'03161000': 'Jefferson', '03164000': 'Galax', '03479000': 'Sugar Grove'}
    policies = [('strict_l168', 'Strict L168', '#bf5700', ''),
                ('combined_2', 'Short gaps ≤2 total hours', '#386f87', ''),
                ('natural_l24', 'Separate natural L24', '#5b8c5a', '//'),
                ('short2_or_l24', 'Short gaps OR L24 route', '#6b4c9a', '..')]
    split = {r['gauge_id']: r for r in report['split_coverage']
             if r['split'] == 'candidate_validation'}
    high = {r['gauge_id']: r for r in report['fold_high_flow']
            if r['fold'] == 'final_development'}
    for row in route['split_coverage']:
        split[row['gauge_id']]['policies']['short2_or_l24'] = row['policies']['short2_or_l24']
    for row in route['fold_high_flow']:
        high[row['gauge_id']]['policies']['short2_or_l24'] = row['policies']['short2_or_l24']
    metrics = [('Potential forecast availability', split, 'forecast_availability_fraction_by_lead'),
               ('Forecast + valid observed target', split, 'evaluable_fraction_all_origins_by_lead'),
               ('Known high-flow target coverage', high, 'known_high_coverage_fraction_by_lead')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.3), sharey=True)
    x = np.arange(3)
    plotted = {}
    for ax, (title, source, key) in zip(axes, metrics):
        plotted[title] = {}
        for i, (policy, label, color, hatch) in enumerate(policies):
            values = []
            for gauge in names:
                series = source[gauge]['policies'][policy][key]
                assert len(series) == 18 and all(v is not None and 0 <= v <= 1 for v in series)
                values.append(100 * min(series))
            plotted[title][policy] = values
            ax.bar(x + (i-1.5)*.19, values, .18, color=color, hatch=hatch,
                   edgecolor='white', linewidth=.4, label=label)
        ax.axhline(90, color='#333333', linestyle='--', linewidth=1)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_xticks(x, list(names.values()), rotation=20, ha='right')
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 90, 100])
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.16)
    axes[0].set_ylabel('Minimum coverage across leads 1–18 (%)')
    fig.suptitle('Development coverage: what short Q gaps can recover', fontsize=17, y=.98)
    fig.text(.5, .92, '2023-07-31 to 2024-07-31 UTC · unchanged observed-target quality',
             ha='center', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .875),
               ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.72, bottom=.32, left=.065, right=.99, wspace=.14)
    fig.text(.04, .12,
             'Dashed line: 90% goal, not an established outcome. All policies retain weather, soil and calendar requirements.\n'
             'First two panels use all requested origins; last uses approved targets above gradient-training q95.\n'
             'Combined route is a structural upper bound pending filler/model qualification; it changes the information used.\n'
             'Counts describe potential input support, not generated forecasts or forecast skill; repeated windows are not independent events.',
             fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    args.preview.mkdir(parents=True, exist_ok=True)
    for suffix, folder in [('svg', args.output), ('png', args.preview)]:
        path = folder / ('gap-recovery-frontier.' + suffix)
        assert not path.exists(), 'Fresh output required'
        fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)
    receipt = {'schema': 'hydra-gap-recovery-figure/v1',
               'source_sha256': hashlib.sha256(raw).hexdigest(),
               'route_report_sha256': hashlib.sha256(route_raw).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'minimum_lead_coverage_percent': plotted,
               'matplotlib_version': matplotlib.__version__,
               'scope': 'DEVELOPMENT_POTENTIAL_COVERAGE_NOT_FORECAST_SKILL'}
    path = args.output / 'sources.json'
    assert not path.exists(), 'Fresh output required'
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
