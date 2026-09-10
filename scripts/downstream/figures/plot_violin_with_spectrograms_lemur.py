#!/usr/bin/env python3
"""
scripts/downstream/figures/plot_violin_with_spectrograms_lemur.py
Lemur figure formatted for MEE:
    (a) Top:    Confidence change violin (single violin, all roar calls)
                Points: (i)(ii) most degraded, (iii)(iv) least degraded
    (b) Bottom left:  Two most degraded calls — Original | Compressed
    (c) Bottom right: Two least degraded calls — Original | Compressed
Usage:
    python scripts/downstream/figures/plot_violin_with_spectrograms_lemur.py
"""
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import PATHS

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS_DIR      = Path(PATHS['results_dir'])
SOURCE_DIR       = Path(PATHS['source_dir'])
RECON_ROOT       = Path(PATHS['recon_dir'])
FIGS_DIR         = Path(PATHS['figures_dir']) / 'downstream'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

LEMUR_SCORES_DIR = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                   'on_original' / 'lemur' / 'raw'
LEMUR_META       = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                   'embeddings' / 'lemur' / 'lemur_all_metadata.csv'
LEMUR_EMB_DIR    = RESULTS_DIR / 'downstream' / 'transfer_learned' / \
                   'embeddings' / 'lemur'
LEMUR_AUDIO_DIR  = SOURCE_DIR / 'black_and_white_ruffed_lemur' / 'Audio3'
LEMUR_ANN_DIR    = SOURCE_DIR / 'black_and_white_ruffed_lemur' / 'Annotations3'

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CODEC          = 'encodec'
BITRATE        = 24.0
LEMUR_SR       = 48000
CODEC_COLOR    = '#08306b'
ROBUST_COLOR   = '#2ca02c'
DEGRADED_COLOR = '#d62728'
MIN_ORIG_CONF  = 0.05
N_FFT          = 4096
HOP_LENGTH     = 512
CONTEXT_SEC    = 2.0
FS             = 22
FS_SMALL       = 20

# ---------------------------------------------------------------------------
# SVL PARSER
# ---------------------------------------------------------------------------
def parse_svl_annotations(svl_path, native_sr=LEMUR_SR):
    tree = ET.parse(str(svl_path))
    root = tree.getroot()
    rows = []
    for point in root.iter('point'):
        label    = point.get('label', '').strip().lower()
        if label not in ('roar', 'no-roar'):
            continue
        frame    = int(point.get('frame', 0))
        duration = int(point.get('duration', 0))
        value    = float(point.get('value', 0))
        extent   = float(point.get('extent', 0))
        rows.append({
            'start_sec': frame / native_sr,
            'end_sec'  : (frame + duration) / native_sr,
            'low_freq' : value,
            'high_freq': value + extent,
            'label'    : label,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# AUDIO HELPERS
# ---------------------------------------------------------------------------
def load_snippet(path, start, end, target_sr):
    info  = sf.info(str(path))
    sr_in = info.samplerate
    s_fr  = max(0, int(round(start * sr_in)))
    e_fr  = min(int(info.frames), int(round(end * sr_in)))
    audio, _ = sf.read(str(path), start=s_fr, frames=e_fr - s_fr,
                       always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr_in != target_sr:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=target_sr)
    return audio

def compute_spec(audio, sr):
    S    = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr,
                                   hop_length=HOP_LENGTH)
    return S_db, freqs, times

def get_audio_lemur(row, load_start, load_end):
    filename = row['filename']
    stem     = Path(filename).stem
    orig_audio  = load_snippet(LEMUR_AUDIO_DIR / filename,
                               load_start, load_end, LEMUR_SR)
    recon_audio = load_snippet(
        RECON_ROOT / 'lemur' / 'lemur_swift2' / CODEC / str(BITRATE) /
        f'{stem}_reconstructed.wav',
        load_start, load_end, LEMUR_SR)
    return orig_audio, recon_audio

def get_annotation_lemur(row):
    filename     = row['filename']
    window_start = float(row['window_start'])
    window_end   = window_start + 3.0
    stem         = Path(filename).stem
    svl_path     = LEMUR_ANN_DIR / f'{stem}.svl'
    if svl_path.exists():
        try:
            anns      = parse_svl_annotations(svl_path)
            roar_anns = anns[
                (anns['label'] == 'roar') &
                (anns['start_sec'] < window_end) &
                (anns['end_sec']   > window_start)
            ]
            if not roar_anns.empty:
                r = roar_anns.iloc[0]
                return (float(r['start_sec']), float(r['end_sec']),
                        float(r['low_freq']),  float(r['high_freq']))
            else:
                print(f"  [FALLBACK] No roar annotation in {stem}.svl")
        except Exception as e:
            print(f"  [ERROR] SVL parse error: {e}")
    else:
        print(f"  [FALLBACK] SVL not found: {svl_path}")
    return window_start, window_end, 300.0, 1400.0

def draw_annotation(ax, ann_start, ann_end, low_freq, high_freq):
    ax.add_patch(plt.Rectangle(
        (ann_start, low_freq),
        ann_end - ann_start,
        high_freq - low_freq,
        linewidth=3.0, edgecolor='#39ff14',
        facecolor='none', linestyle='--', zorder=5))

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------
def load_call_df():
    print("Loading Lemur data...")
    meta     = pd.read_csv(LEMUR_META)
    test_idx = np.load(LEMUR_EMB_DIR / 'lemur_temporal_test_indices.npy')
    meta     = meta.iloc[test_idx].reset_index(drop=True)
    meta     = meta.reset_index().rename(columns={'index': 'clip_idx'})

    scores_orig = pd.read_csv(
        LEMUR_SCORES_DIR / 'scores_original_on_original.csv')
    scores_enc  = pd.read_csv(
        LEMUR_SCORES_DIR / f'scores_original_on_{CODEC}{BITRATE}.csv')

    roar_clips = meta[meta['label'] == 1].copy()
    call_df = roar_clips.merge(
        scores_orig[['clip_idx', 'score']].rename(
            columns={'score': 'conf_orig'}),
        on='clip_idx', how='left')
    call_df = call_df.merge(
        scores_enc[['clip_idx', 'score']].rename(
            columns={'score': 'conf_enc'}),
        on='clip_idx', how='left')
    call_df[['conf_orig', 'conf_enc']] = \
        call_df[['conf_orig', 'conf_enc']].fillna(0.0)
    call_df['drop_enc'] = call_df['conf_enc'] - call_df['conf_orig']
    print(f"  Roar calls: {len(call_df)}")
    return call_df

# ---------------------------------------------------------------------------
# SELECT EXAMPLES
# ---------------------------------------------------------------------------
def select_examples(call_df):
    df = call_df[call_df['conf_orig'] >= MIN_ORIG_CONF].copy()
    if len(df) < 4:
        df = call_df.copy()

    most_deg  = df.nsmallest(2, 'drop_enc').copy().reset_index(drop=True)
    least_deg = df.nlargest(2,  'drop_enc').copy().reset_index(drop=True)

    most_deg['label']  = ['i',   'ii']
    most_deg['color']  = [DEGRADED_COLOR, DEGRADED_COLOR]
    least_deg['label'] = ['iii', 'iv']
    least_deg['color'] = [ROBUST_COLOR,   ROBUST_COLOR]

    examples = pd.concat([most_deg, least_deg]).reset_index(drop=True)
    return examples

# ---------------------------------------------------------------------------
# RENDER SPECTROGRAM PAIR
# ---------------------------------------------------------------------------
def render_spectrogram_pair(fig, gs_title, gs_orig, gs_recon,
                            ex_row, is_last_row=False):
    lbl  = ex_row['label']
    drop = float(ex_row['drop_enc'])

    ax_title = fig.add_subplot(gs_title)
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, f'({lbl})',
                  transform=ax_title.transAxes,
                  fontsize=FS, color='black',
                  ha='center', va='center')

    ax_orig  = fig.add_subplot(gs_orig)
    ax_recon = fig.add_subplot(gs_recon, sharey=ax_orig)

    ann_start, ann_end, low_freq, high_freq = get_annotation_lemur(ex_row)
    load_start = max(0.0, ann_start - CONTEXT_SEC)
    load_end   = ann_end + CONTEXT_SEC

    pad  = (high_freq - low_freq) * 2
    ymin = max(0, low_freq - pad)
    ymax = min(LEMUR_SR // 2, high_freq + pad)

    try:
        orig_audio, recon_audio = get_audio_lemur(ex_row, load_start, load_end)
        if orig_audio is None or recon_audio is None:
            raise ValueError("Could not load audio")

        min_len     = min(len(orig_audio), len(recon_audio))
        orig_audio  = orig_audio[:min_len]
        recon_audio = recon_audio[:min_len]

        if len(recon_audio) == 0 or np.all(recon_audio == 0):
            print(f"  Warning: reconstructed audio for ({lbl}) zeros/empty")
            ax_orig.set_visible(False)
            ax_recon.set_visible(False)
            return

        spec_orig,  freqs, times = compute_spec(orig_audio,  LEMUR_SR)
        spec_recon, _,     _     = compute_spec(recon_audio, LEMUR_SR)
        times_abs = times + load_start

        vmin = np.percentile(spec_orig, 10)
        vmax = np.percentile(spec_orig, 99)

        for ax, spec in [(ax_orig, spec_orig), (ax_recon, spec_recon)]:
            ax.imshow(spec, aspect='auto', origin='lower',
                      extent=[times_abs[0], times_abs[-1],
                              freqs[0], freqs[-1]],
                      cmap='magma', vmin=vmin, vmax=vmax,
                      interpolation='bilinear')
            draw_annotation(ax, ann_start, ann_end, low_freq, high_freq)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(times_abs[0], times_abs[-1])
            ax.tick_params(labelsize=FS_SMALL)
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        ax_orig.set_title(
            f'Original\n(conf = {ex_row["conf_orig"]:.2f})',
            fontsize=FS, pad=10)
        ax_recon.set_title(
            f'Reconstructed\n'
            f'(conf = {ex_row["conf_enc"]:.2f}, '
            f'$\\Delta$ = {drop:+.2f})',
            fontsize=FS, pad=10)
        ax_orig.set_ylabel('Freq (Hz)', fontsize=FS)
        plt.setp(ax_recon.get_yticklabels(), visible=False)
        if is_last_row:
            ax_orig.set_xlabel('Time (s)', fontsize=FS)
            ax_recon.set_xlabel('Time (s)', fontsize=FS)

    except Exception as e:
        print(f"  Could not load audio for ({lbl}): {e}")
        ax_orig.set_visible(False)
        ax_recon.set_visible(False)

# ---------------------------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------------------------
def plot_lemur(call_df, examples):
    fig = plt.figure(figsize=(20.0, 18.0))
    plt.rcParams['font.family'] = 'Arial'

    outer_gs = gridspec.GridSpec(2, 1,
                                 height_ratios=[1.0, 2.8],
                                 hspace=0.30)

    # ---- (a) VIOLIN ----
    ax_violin = fig.add_subplot(outer_gs[0, 0])

    parts = ax_violin.violinplot(
        call_df['drop_enc'].values,
        positions=[0],
        orientation='horizontal',
        showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(CODEC_COLOR)
        pc.set_alpha(0.55)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.4)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.8)
    parts['cmedians'].set_zorder(5)

    ax_violin.axvline(0, color='black', linestyle='-', lw=0.9, alpha=0.4)
    ax_violin.axvspan(-1.05, 0, alpha=0.03, color='red')
    ax_violin.axvspan(0, 1.05, alpha=0.03, color='green')

    for _, ex_row in examples.iterrows():
        lbl   = ex_row['label']
        color = ex_row['color']
        val   = float(ex_row['drop_enc'])
        ax_violin.scatter(val, 0, marker='o', s=150,
                          color=color, edgecolors='black',
                          linewidths=1.2, zorder=15)
        offset = -0.18 if lbl in ['i', 'ii'] else 0.18
        va     = 'top'  if lbl in ['i', 'ii'] else 'bottom'
        # ax_violin.text(val, offset, lbl,
        #                transform=ax_violin.get_xaxis_transform(),
        #                fontsize=FS, color='black',
        #                ha='center', va=va, zorder=16)
        ax_violin.text(val, offset, lbl,
               fontsize=FS, color='black',
               ha='center', va=va, zorder=16)

        # And update offsets to data coordinates (small fraction of y range):
        offset = -0.04 if lbl in ['i', 'ii'] else 0.04
        va     = 'top'  if lbl in ['i', 'ii'] else 'bottom'

    ax_violin.set_yticks([0])
    ax_violin.set_yticklabels(['Roar calls'], fontsize=FS_SMALL)
    ax_violin.set_xlabel('Confidence change (compressed \u2212 original)',
                         fontsize=FS, labelpad=9)
    ax_violin.set_xlim(-1.05, 1.05)
    ax_violin.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_violin.tick_params(axis='x', labelsize=FS_SMALL)
    ax_violin.grid(True, axis='x', alpha=0.3)
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)
    ax_violin.text(0.0, 1.05,
                   '(a) Confidence change distribution (EnCodec 24.0 kbps)',
                   transform=ax_violin.transAxes,
                   fontsize=FS, va='bottom', ha='left')

    # ---- (b) & (c) SPECTROGRAMS ----
    # bottom_gs = gridspec.GridSpecFromSubplotSpec(
    #     1, 2, subplot_spec=outer_gs[1, 0], wspace=0.18)
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[1, 0], wspace=0.35)

    PANEL_TAGS = [
        '(b) Calls with highest confidence loss',
        '(c) Calls with highest confidence gain',
    ]
    GROUP_EXAMPLES = [
        examples[examples['label'].isin(['i',   'ii'])].reset_index(drop=True),
        examples[examples['label'].isin(['iii', 'iv'])].reset_index(drop=True),
    ]

    for col_idx, (group_examples, panel_tag) in enumerate(
            zip(GROUP_EXAMPLES, PANEL_TAGS)):

        group_sub_gs = gridspec.GridSpecFromSubplotSpec(
            5, 2, subplot_spec=bottom_gs[0, col_idx],
            height_ratios=[0.12, 0.12, 1.0, 0.12, 1.0],
            hspace=0.50, wspace=0.10)

        ax_panel = fig.add_subplot(group_sub_gs[0, :])
        ax_panel.axis('off')
        ax_panel.text(0.5, 0.5, panel_tag,
                      transform=ax_panel.transAxes,
                      fontsize=FS, color='black',
                      va='center', ha='center')

        for row_in_group in range(2):
            ex_row    = group_examples.iloc[row_in_group]
            title_row = 1 if row_in_group == 0 else 3
            spec_row  = 2 if row_in_group == 0 else 4
            is_last   = (row_in_group == 1)

            render_spectrogram_pair(
                fig=fig,
                gs_title=group_sub_gs[title_row, :],
                gs_orig=group_sub_gs[spec_row,   0],
                gs_recon=group_sub_gs[spec_row,  1],
                ex_row=ex_row,
                is_last_row=is_last)

    output_path = FIGS_DIR / 'violin_with_spectrograms_lemur.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    call_df  = load_call_df()
    examples = select_examples(call_df)

    print("\n  Examples selected:")
    for _, r in examples.iterrows():
        print(f"    ({r['label']}) {r['filename']}  "
              f"window={r['window_start']:.1f}s  "
              f"drop={r['drop_enc']:+.3f}  "
              f"orig={r['conf_orig']:.2f}  enc={r['conf_enc']:.2f}")

    plot_lemur(call_df, examples)

if __name__ == '__main__':
    main()