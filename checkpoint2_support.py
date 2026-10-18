"""Reusable, outcome-independent 48-hour feature extraction for Checkpoints 2/3."""
from pathlib import Path
import numpy as np
import pandas as pd

STATIC = ['Age', 'Gender', 'Height', 'ICUType']
PARAMETERS = ['ALP', 'ALT', 'AST', 'Age', 'Albumin', 'BUN', 'Bilirubin',
              'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Gender',
              'Glucose', 'HCO3', 'HCT', 'HR', 'Height', 'ICUType', 'K', 'Lactate',
              'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na',
              'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp',
              'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH']

def build_features(folder):
    rows, seen = [], set()
    files = sorted(Path(folder).glob('*.txt'))
    if not files:
        raise ValueError(f'No patient files in {folder}')
    for path in files:
        frame = pd.read_csv(path, dtype={'Time': 'string', 'Value': float})
        ids = frame.loc[frame.Parameter.eq('RecordID'), 'Value']
        assert len(ids) == 1 and ids.iloc[0] == int(path.stem)
        rid = int(ids.iloc[0])
        assert rid not in seen
        seen.add(rid)
        frame = frame.loc[frame.Parameter.ne('RecordID')].copy()
        assert frame.Time.str.fullmatch(r'\d{2}:\d{2}').all()
        parts = frame.Time.str.split(':', expand=True).astype(int)
        assert parts[1].between(0, 59).all()
        frame['minutes'] = parts[0] * 60 + parts[1]
        assert frame.minutes.between(0, 2880).all()
        frame['clean'] = frame.Value.mask(frame.Value.eq(-1))
        frame = frame.sort_values('minutes', kind='stable')
        row = {'RecordID': rid}
        for name in PARAMETERS:
            if name in STATIC:
                row[name] = np.nan
            else:
                row[name + '_count'] = 0
                for stat in ['last', 'min', 'max', 'mean']:
                    row[name + '_' + stat] = np.nan
        assert set(frame.Parameter).issubset(PARAMETERS), 'Unexpected source parameter'
        for name, group in frame.groupby('Parameter', sort=False):
            valid = group.clean.dropna()
            if name in STATIC:
                row[name] = valid.iloc[0] if len(valid) else np.nan
            else:
                row[name + '_count'] = len(valid)
                for stat in ['last', 'min', 'max', 'mean']:
                    row[name + '_' + stat] = (valid.iloc[-1] if stat == 'last' else getattr(valid, stat)()) if len(valid) else np.nan
        rows.append(row)
    result = pd.DataFrame(rows).set_index('RecordID').sort_index()
    counts = [c for c in result if c.endswith('_count')]
    result[counts] = result[counts].fillna(0)
    # MechVent records contain only 1s; retain recording frequency, not constant value summaries.
    result = result.drop(columns=[c for c in result if c.startswith('MechVent_') and c != 'MechVent_count'])
    # One flag per source parameter; all four summaries share the same missingness.
    missing = {name + '_missing': result[name].isna().astype(int)
               for name in result if name in ['Age', 'Height'] or name.endswith('_last')}
    return pd.concat([result, pd.DataFrame(missing, index=result.index)], axis=1)
