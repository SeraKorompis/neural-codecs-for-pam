#!/usr/bin/env python3
"""Four numbers for the BirdNET paragraph (EnCodec 6 kbps by default).
Usage: python scripts/downstream/figures/birdnet_text_numbers_short.py [tag]
tag examples: encodec_6.0 (default), dac_5.33, encodec_24.0, dac_8.0
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_auc_scatter_all import load_and_pivot_data, BIRDNET_RES_DIR

tag = sys.argv[1] if len(sys.argv) > 1 else 'encodec_6.0'
path = next(BIRDNET_RES_DIR / n for n in ['per_species_ovr.csv', 'per_species.csv']
            if (BIRDNET_RES_DIR / n).exists())
sp = load_and_pivot_data(path, 'birdnet')
if 'species' in sp.columns:
    sp = sp.set_index('species')

o, c = sp['auc_orig'], sp[f'auc_{tag}']
pct = (o - c) / o * 100

print(f'1. AUC on raw audio across {len(sp)} species: {o.min():.2f} to {o.max():.2f}')
print(f'2. {tag} drop across species: {pct.min():.1f}% to {pct.max():.1f}%')

q = sp.index[o >= o.quantile(0.75)]          # upper quartile, as in the figure key
print('3. Least degraded (upper quartile of raw AUC):')
for s in pct[q].sort_values().index[:5]:
    print(f'   {s}: {pct[s]:.1f}% drop ({o[s]:.2f} -> {c[s]:.2f})')
print('4. Most degraded (upper quartile of raw AUC):')
for s in pct[q].sort_values(ascending=False).index[:5]:
    print(f'   {s}: {o[s]:.2f} -> {c[s]:.2f} ({pct[s]:.1f}% drop)')
