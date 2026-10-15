"""Audit the October 14 revision from saved results; no fitting or manuscript edits."""
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METRICS = ['relation_w1', 'c2st', 'rf', 'membership', 'clinical_tv']
KEYS = ['size', 'seed', 'variant', 'processing']


def close(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True), (a, b)


def main():
    controls = pd.read_csv(ROOT / 'runs/iteration14_controls/controls.csv')
    assert len(controls) == 560
    assert not controls.duplicated(['study', 'dataset', *KEYS]).any()
    assert controls[METRICS].notna().all().all()
    registry_counts = {}
    for study in ('iteration13', 'iteration14'):
        base = pd.read_csv(ROOT / f'reports/{study}/confirm_all.csv')
        base = base[(base.status == 'complete') & base['size'].isin([250, 1000, 2000])]
        extra = controls[controls.study == study]
        expected = pd.concat([base, extra], ignore_index=True)
        actual = pd.read_csv(ROOT / f'reports/final/registry_{study}.csv')
        keys = ['dataset', *KEYS] if study == 'iteration14' else KEYS
        assert len(expected) == len(actual)
        assert not expected.duplicated(keys).any() and not actual.duplicated(keys).any()
        left, right = expected.set_index(keys).sort_index(), actual.set_index(keys).sort_index()
        assert left.index.equals(right.index)
        close(left[METRICS].to_numpy(float), right[METRICS].to_numpy(float))
        registry_counts[study] = dict(rows=len(actual), added_controls=len(extra))

    registry = pd.read_csv(ROOT / 'reports/final/registry_iteration14.csv')
    published = pd.read_csv(ROOT / 'reports/final/per_dataset.csv')
    cells, paired = [], []
    for r in published.itertuples():
        cell = registry[(registry.dataset == r.dataset) & (registry['size'] == r.size)]
        mine = cell[(cell.variant == 'mhd:c1') & (cell.processing == 'primary')].set_index('seed')
        means = cell.groupby(['variant', 'processing'])[METRICS].mean()
        comp = means[[not name.startswith('mhd') and name != 'bootstrap' for name in means.index.get_level_values(0)]]
        eligible = comp[comp.membership <= .60]
        ours_eligible = bool(mine.membership.mean() <= .60)
        assert ours_eligible == r.ours_eligible
        cell_evidence = dict(dataset=r.dataset, size=r.size, ours_eligible=ours_eligible)
        for metric, pool, ours_name, other_name, wins_name, lower in (
            ('relation_w1', comp, 'ours_relation', 'control_relation', 'relation_wins', True),
            ('c2st', eligible, 'ours_c2st', 'comparator_c2st', 'c2st_wins', True),
            ('rf', eligible, 'ours_rf', 'comparator_rf', 'rf_wins', False),
            ('clinical_tv', eligible, 'ours_category', 'comparator_category', 'category_wins', True),
        ):
            arm = pool[metric].idxmin() if lower else pool[metric].idxmax()
            other = cell[(cell.variant == arm[0]) & (cell.processing == arm[1])].set_index('seed')
            assert set(mine.index) == set(other.index)
            difference = mine[metric] - other[metric]
            wins = int((difference < 0).sum() if lower else (difference > 0).sum())
            close(mine[metric].mean(), getattr(r, ours_name))
            close(pool.loc[arm, metric], getattr(r, other_name))
            assert f'{wins}/{len(difference)}' == getattr(r, wins_name)
            cell_evidence[metric] = dict(ours=float(mine[metric].mean()), comparator=float(pool.loc[arm, metric]),
                                        arm=list(arm), wins=wins, n=len(difference), median_difference=float(difference.median()))
            if metric == 'c2st':
                best = ours_eligible and mine[metric].mean() < pool.loc[arm, metric]
                assert best == r.ours_best_eligible
                cell_evidence['ours_best_eligible'] = bool(best)
                paired.extend(dict(dataset=r.dataset, size=r.size, seed=int(seed), difference=float(value))
                              for seed, value in difference.items())
        cells.append(cell_evidence)

    prior = json.loads((ROOT / 'reports/publication_review_2026-10-13/audit_evidence.json').read_text())
    control = controls[(controls.study == 'iteration14') & (controls.dataset == 'nhanes_chem_l') &
                       (controls['size'] == 250) & (controls.variant == 'copula') & (controls.processing == 'textbook')].set_index('seed')
    for row in prior['canonical_copula_rows']:
        close(control.loc[row['seed'], 'membership'], row['membership_canonical'])
        close(control.loc[row['seed'], 'c2st'], row['c2st_canonical'])

    snapshots = {}
    for prefix in ('890ebbee', '1003bc8c', '1626362e'):
        directory = next((ROOT / 'reproducibility/source_snapshots').glob(prefix + '*'))
        digest = hashlib.sha256()
        for path in sorted(directory.glob('*.py')):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
        assert digest.hexdigest() == directory.name
        snapshots[prefix] = digest.hexdigest()

    archive = ROOT / 'release/medhyperdiff-paper-release.zip'
    with zipfile.ZipFile(archive) as bundle:
        prefix = 'medhyperdiff-paper-release/'
        for name in ('paper/ieee/main.pdf', 'paper/ieee/main.tex', 'src/medhyperdiff/relations.py',
                     'scripts/report_final.py', 'scripts/controls_iteration14.py'):
            assert bundle.read(prefix + name) == (ROOT / name).read_bytes()
        missing_control_csv = prefix + 'runs/iteration14_controls/controls.csv' not in bundle.namelist()
    output = dict(pdf_sha256=hashlib.sha256((ROOT / 'paper/ieee/main.pdf').read_bytes()).hexdigest(),
                  controls=len(controls), registry_counts=registry_counts, final_table_cells_verified=len(cells),
                  ours_best_eligible=sum(c['ours_best_eligible'] for c in cells), cells=cells,
                  previous_independent_canonical_control_audit_matches=True,
                  source_snapshots_verified=snapshots, release_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  release_selected_files_match_current=True, release_missing_report_input_control_csv=missing_control_csv,
                  scope='Saved-result consistency audit. No generator, metric classifier or full test suite was rerun.')
    (OUT / 'audit_evidence.json').write_text(json.dumps(output, indent=2) + '\n')
    pd.DataFrame(paired).to_csv(OUT / 'paired_realism_differences.csv', index=False)
    print(json.dumps({k:v for k,v in output.items() if k != 'cells'}, indent=2))
    print('Narrow CBC win:', next(c for c in cells if c['dataset']=='nhanes_cbc_l' and c['size']==1000)['c2st'])


if __name__ == '__main__':
    main()
