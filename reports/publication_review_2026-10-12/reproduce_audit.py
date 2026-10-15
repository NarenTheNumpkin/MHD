import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
parsed = ast.parse((ROOT / 'src/medhyperdiff/data.py').read_text())
columns = {}
for node in parsed.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('NHANES_COLUMNS', 'CBC_COLUMNS', 'CHEM_COLUMNS'):
                columns[target.id] = ast.literal_eval(node.value)


def cohort(panel, cycle):
    cols = columns[{'cbc': 'CBC_COLUMNS', 'chem': 'CHEM_COLUMNS', 'original': 'NHANES_COLUMNS'}[panel]]
    raw = ROOT / 'data/raw/nhanes'
    read = lambda name: pd.read_sas(raw / f'{name}_{cycle}.xpt', format='xport')
    labs = [c for c in cols if c.startswith('LB')]
    frame = read('DEMO')[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
    frame = frame.merge(read('CBC' if panel == 'cbc' else 'BIOPRO')[['SEQN'] + labs], on='SEQN')
    frame = frame.merge(read('DIQ')[['SEQN', 'DIQ010']], on='SEQN')
    frame = frame[(frame.RIDAGEYR >= 20) & frame.DIQ010.isin([1, 2]) & frame[labs].notna().any(axis=1)]
    name = 'nhanes' if panel == 'original' else ('nhanes_cbc' if panel == 'cbc' and cycle == 'J' else f'nhanes_{panel}_{cycle.lower()}')
    saved = pd.read_csv(ROOT / 'data/processed' / f'{name}.csv')
    assert len(frame) == len(saved)
    for c in labs:
        assert np.allclose(frame[c], saved[cols[c]], rtol=1e-12, atol=1e-12, equal_nan=True), (name, c)
    assert np.allclose(frame.RIDAGEYR, saved.age)
    return frame.SEQN.to_numpy()


out = {'cohort_overlap': [], 'fit_sizes_and_overlap': [], 'cells': []}
for cycle, a, b in [('J', 'original', 'cbc'), ('I', 'cbc', 'chem'), ('L', 'cbc', 'chem')]:
    x, y = cohort(a, cycle), cohort(b, cycle)
    item = {'cycle': cycle, 'panels': [a, b], 'rows': [len(x), len(y)], 'shared_people': len(set(x) & set(y))}
    out['cohort_overlap'].append(item)
    print('cohort_overlap', item, flush=True)

plan = json.loads((ROOT / 'configs/iteration14_splits.json').read_text())
for name, value in plan['datasets'].items():
    if value['role'] != 'confirmation':
        continue
    for n in [250, 1000, 2000]:
        samples = [entry for key, entry in value['samples'].items() if key.startswith(f'n{n}_')]
        if not samples:
            continue
        fits = [set(entry['fit']) for entry in samples]
        overlaps = [len(a & b) / min(len(a), len(b)) for a, b in itertools.combinations(fits, 2)]
        assert all(not (fit & set(value['reference'])) for fit in fits)
        item = {'dataset': name, 'labelled_n': n, 'fit_sizes': sorted(set(map(len, fits))),
                'fit_pair_overlap_min': min(overlaps), 'fit_pair_overlap_max': max(overlaps), 'reference': len(value['reference'])}
        out['fit_sizes_and_overlap'].append(item)

all_rows = pd.read_csv(ROOT / 'reports/iteration14/confirm_all.csv')
d = all_rows[all_rows['size'].isin([250, 1000, 2000])]
out['jobs'] = {'confirm_all': int(all_rows.id.nunique()), 'analysed_confirmation': int(d.id.nunique()),
               'omitted_n100': int(all_rows[all_rows['size'] == 100].id.nunique())}
out['metric_reconciliation'] = {'jobs_checked': 0, 'metric_hashes_available': 0, 'hash_mismatches': [], 'value_mismatches': []}
audit = out['metric_reconciliation']
for run_id, group in d.groupby('id'):
    folder = ROOT / 'runs/iteration14' / run_id
    manifest = json.loads((folder / 'manifest.json').read_text())
    metrics_bytes = (folder / 'metrics.json').read_bytes()
    metrics = json.loads(metrics_bytes)
    audit['jobs_checked'] += 1
    if 'metrics_sha256' in manifest:
        audit['metric_hashes_available'] += 1
        if hashlib.sha256(metrics_bytes).hexdigest() != manifest['metrics_sha256']:
            audit['hash_mismatches'].append(run_id)
    for _, row in group.iterrows():
        for key in ['c2st', 'relation_w1']:
            if not np.isclose(row[key], metrics[row.processing][key], rtol=1e-12, atol=1e-12):
                audit['value_mismatches'].append([run_id, row.processing, key])

for (dataset, n), cell in d.groupby(['dataset', 'size']):
    m = cell.groupby(['variant', 'processing'])[['rf', 'rf_real', 'c2st', 'membership', 'clinical_tv', 'relation_w1']].mean()
    ours = m.loc[('mhd:c1', 'primary')]
    comp = m[(~m.index.get_level_values(0).str.startswith('mhd')) & (m.index.get_level_values(0) != 'bootstrap') & (m.membership <= .60)]
    published = comp[~comp.index.get_level_values(0).str.endswith('_rp')]
    best_c2st, best_rf = comp.c2st.idxmin(), comp.rf.idxmax()
    item = {'dataset': dataset, 'size': int(n), 'ours': ours.to_dict(),
            'best_c2st_arm': list(best_c2st), 'best_c2st': float(comp.c2st.min()),
            'best_utility_arm': list(best_rf), 'best_utility': float(comp.rf.max()),
            'best_published_c2st_arm': list(published.c2st.idxmin()),
            'best_clinical_tv': float(comp.clinical_tv.min()),
            'clinical_best_arm': list(comp.clinical_tv.idxmin())}
    out['cells'].append(item)
out['pipeline_summary'] = pd.read_csv(ROOT / 'reports/iteration12/a7_pipeline_summary.csv').to_dict('records')
(OUT / 'recomputed_audit.json').write_text(json.dumps(out, indent=2))
print('fit_sizes_and_overlap', json.dumps(out['fit_sizes_and_overlap'], indent=2), flush=True)
print('jobs', out['jobs'], 'metric_reconciliation', audit, flush=True)
print('cell_comparisons', json.dumps(out['cells'], indent=2), flush=True)
print('pipeline_summary', json.dumps(out['pipeline_summary'], indent=2), flush=True)
