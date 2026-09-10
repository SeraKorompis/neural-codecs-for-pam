import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/rds/general/user/stk25/home/ai_audio_compression/project')
from src.config import PATHS

results_dir = Path(PATHS['results_dir'])
df = pd.read_csv(results_dir / 'reconstruction_quality' / 'metrics_merged.csv')

# ---------------------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------------------
MP3_KEEP  = [8.0, 16.0, 24.0]
OPUS_KEEP = [8.0, 14.0, 24.0]
df = df[
    ~((df['codec'] == 'mp3')  & (~df['bitrate'].isin(MP3_KEEP))) &
    ~((df['codec'] == 'opus') & (~df['bitrate'].isin(OPUS_KEEP)))
]

# DAC nominal kbps
DAC_KBPS_MAP = {2: 1.78, 3: 2.67, 6: 5.33, 9: 8.0}
df['kbps'] = df.apply(
    lambda r: DAC_KBPS_MAP[int(r['bitrate'])] if r['codec'] == 'dac'
              else float(r['bitrate']), axis=1)

# ---------------------------------------------------------------------------
# RTF — neural codecs GPU only, conventional codecs CPU
# ---------------------------------------------------------------------------
neural = df[(df['codec'].isin(['encodec', 'dac'])) & (df['device'] == 'cuda')].copy()
conv   = df[df['codec'].isin(['mp3', 'opus'])].copy()
df_rtf = pd.concat([neural, conv], ignore_index=True)

# RTF = audio_duration / processing_time (higher = faster)
df_rtf['encode_rtf']   = df_rtf['audio_duration_sec'] / df_rtf['encode_time_sec']
df_rtf['decode_rtf']   = df_rtf['audio_duration_sec'] / df_rtf['decode_time_sec']
df_rtf['combined_rtf'] = df_rtf['audio_duration_sec'] / df_rtf['compression_time_sec']

rtf_agg = (
    df_rtf.groupby(['codec', 'bitrate', 'kbps'])
    .agg(
        encode_rtf_mean   = ('encode_rtf',   'mean'),
        decode_rtf_mean   = ('decode_rtf',   'mean'),
        combined_rtf_mean = ('combined_rtf', 'mean'),
        encode_rtf_std    = ('encode_rtf',   'std'),
        decode_rtf_std    = ('decode_rtf',   'std'),
        combined_rtf_std  = ('combined_rtf', 'std'),
        n_files           = ('filename',     'count'),
    )
    .reset_index()
    .sort_values(['codec', 'kbps'])
)

out = results_dir / 'reconstruction_quality' / 'processing_time_rtf_agg.csv'
rtf_agg.to_csv(out, index=False)
print(f'Saved: {out}')
print(rtf_agg[['codec','kbps','encode_rtf_mean','decode_rtf_mean','combined_rtf_mean','n_files']].to_string())