# scripts/downstream/transfer_learned/merge_northeastern_ovr.py
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '/rds/general/user/stk25/home/ai_audio_compression/project')
from src.config import PATHS

results_dir  = Path(PATHS['results_dir']) / 'downstream' / 'transfer_learned' / \
               'on_original' / 'northeastern'
per_cond_dir = results_dir / 'raw' / 'per_condition'

macro_files = sorted(per_cond_dir.glob('macro_metrics_ovr_*.csv'))
per_sp_files = sorted(per_cond_dir.glob('per_species_ovr_*.csv'))

print(f"Found {len(macro_files)} macro files, {len(per_sp_files)} per-species files")

pd.concat([pd.read_csv(f) for f in macro_files]).to_csv(
    results_dir / 'macro_metrics_ovr.csv', index=False)
print("Saved macro_metrics_ovr.csv")

pd.concat([pd.read_csv(f) for f in per_sp_files]).to_csv(
    results_dir / 'per_species_ovr.csv', index=False)
print("Saved per_species_ovr.csv")