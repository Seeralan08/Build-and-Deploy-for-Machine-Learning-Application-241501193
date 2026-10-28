"""Train/evaluate the offline EV charging demand model.
The dataset is deterministic synthetic teaching data, not field measurements.
Run: python run_analysis.py
"""
from pathlib import Path
import csv, math
import numpy as np
BASE=Path(__file__).parent
rows=list(csv.DictReader((BASE/'data'/'india_ev_charging_demand_synthetic.csv').open(encoding='utf-8')))
FEATURES=['hour','is_weekend','temperature_c','rainfall_mm','charger_ports','event_evening','latitude','longitude']
X=np.array([[float(row[k]) for k in FEATURES] for row in rows]); y=np.array([float(row['sessions']) for row in rows])
split=int(.8*len(y)); mean=X[:split].mean(0); scale=X[:split].std(0); scale[scale==0]=1
Z=np.column_stack([np.ones(len(y)),(X-mean)/scale]); coef=np.linalg.lstsq(Z[:split],y[:split],rcond=None)[0]
pred=Z[split:]@coef; truth=y[split:]; mae=np.mean(np.abs(truth-pred)); rmse=np.sqrt(np.mean((truth-pred)**2)); r2=1-np.sum((truth-pred)**2)/np.sum((truth-truth.mean())**2); baseline=np.mean(np.abs(truth-y[:split].mean()))
print(f'Rows: {len(rows)} | train: {split} | test: {len(rows)-split}')
print(f'Test MAE: {mae:.3f} sessions')
print(f'Test RMSE: {rmse:.3f} sessions')
print(f'Test R2: {r2:.4f}')
print(f'Training-mean baseline MAE: {baseline:.3f} sessions')
