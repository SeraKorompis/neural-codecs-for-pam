# src/figures.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path


# DATASET VISUALISATION CONFIG
# window_sec:   length of spectrogram window to display in seconds
# max_freq:     upper frequency limit for spectrogram display in Hz
# n_fft:        FFT size for spectrogram display, larger = finer frequency resolution
# hop_length:   hop size for spectrogram display, smaller = finer time resolution
# Note: n_fft and hop_length affect visualisation only, not evaluation metrics

DATASET_VIZ_CONFIG = {
    'anuraset'          : {'window_sec': 10,  'max_freq': 8000,  'n_fft': 1024, 'hop_length': 256},
    'northeastern_birds': {'window_sec': 30,  'max_freq': 16000, 'n_fft': 2048, 'hop_length': 512},
    'lemur_s4a'         : {'window_sec': 60,  'max_freq': 2000,  'n_fft': 4096, 'hop_length': 1024},
    'lemur_swift1'      : {'window_sec': 60,  'max_freq': 2000,  'n_fft': 4096, 'hop_length': 1024},
    'lemur_swift2'      : {'window_sec': 60,  'max_freq': 2000,  'n_fft': 4096, 'hop_length': 1024},
    'bee'               : {'window_sec': 30,  'max_freq': 8000,  'n_fft': 1024, 'hop_length': 256},
}


# PAPER BASELINES
# From DAC paper Table 3 — evaluated on speech, music and environmental sounds at 44.1kHz
# Encodec baselines from DAC paper (same evaluation protocol as DAC for fair comparison)
# DAC paper reports SI-SDR not SI-SNR — similar but not identical metrics
# These are reference points only — not directly comparable to bioacoustic results

PAPER_BASELINES = {
    'encodec': {
        'si_snr' : [(1.5, -0.02), (3.0, 2.94), (6.0, 5.99),
                    (12.0, 8.36), (24.0, 9.59)],
        'stft'   : [(1.5, 4.30),  (3.0, 4.19), (6.0, 4.10),
                    (12.0, 4.02), (24.0, 3.97)],
    },
    'dac': {
        'si_snr' : [(1.78, 2.16), (2.67, 4.41), (5.33, 8.13), (8.0, 10.75)],
        'stft'   : [(1.78, 1.95), (2.67, 1.85), (5.33, 1.69), (8.0, 1.60)],
    },
    'opus': {
        'si_snr': [(8.0, 5.68), (14.0, 8.02), (24.0, 11.65)],
        'stft':   [(8.0, 5.72), (14.0, 2.14), (24.0, 1.90)],
    },
}


# SNAC empirical bitrates (based on GitHub repo), fixed per model SR
SNAC_BITRATE_KBPS = {
    24000: 0.98,
    32000: 1.9,
    44100: 2.6,
}

# Note: Encodec bitrates are determined by the bandwidth parameter (see src/compress.py)
# DAC bitrates are determined by the number of codebooks used (see src/compress.py), 
# but is not fixed in the paper thus it is computed empirically 


# HELPERS

def _get_db(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Compute log-magnitude spectrogram."""
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return librosa.amplitude_to_db(S, ref=np.max)


def _overlay_boxes(ax:          plt.Axes,
                   annotations: pd.DataFrame,
                   start_sec:   float,
                   end_sec:     float,
                   max_freq:    int,
                   sr:          int) -> None:
    """
    Overlay annotation bounding boxes on a spectrogram axes.
    For datasets without frequency bounds (low_freq=0, high_freq=Nyquist),
    boxes span full spectrogram height — showing time boundaries only.
    For datasets with frequency bounds, boxes show exact time-frequency extent.
    Draws species label text in top-left corner of each box.
    """
    nyquist = sr // 2
    for _, row in annotations.iterrows():
        if (row['end_sec'] >= start_sec) and (row['start_sec'] <= end_sec):
            vis_start = max(row['start_sec'], start_sec) - start_sec
            vis_end   = min(row['end_sec'],   end_sec)   - start_sec
            low_freq  = float(row.get('low_freq',  0))
            high_freq = float(row.get('high_freq', nyquist))

            # If no frequency bounds — span full spectrogram height
            if low_freq == 0 and high_freq >= nyquist:
                low_freq  = 0
                high_freq = max_freq

            rect = plt.Rectangle(
                (vis_start, low_freq),
                vis_end - vis_start,
                high_freq - low_freq,
                linewidth=2, edgecolor='white',
                facecolor='none', alpha=1.0
            )
            ax.add_patch(rect)

            # Draw label text in top-left corner of box
            label = str(row.get('label', ''))
            if label:
                ax.text(
                    vis_start + 0.02, high_freq * 0.97,
                    label,
                    color='white', fontsize=7, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='black', alpha=0.5,
                              edgecolor='none'),
                )


# FIGURE 1: METRICS VS BITRATE

def plot_metrics_vs_bitrate(metrics:  pd.DataFrame,
                             dataset:  str,
                             figs_dir: Path,
                             title:    str = None) -> None:
    """
    Four-panel line plot of SI-SNR and STFT distance vs bitrate (kbps).
    One line per codec: whole-recording and call-region rows.
    Paper baselines shown as dashed lines for reference.
    For DAC, empirical_kbps is used as x-axis.
    For Encodec, bitrate is used directly (already in kbps).
    Note: paper baselines evaluated on speech/music/env (not bioacoustic data).
    """
    # If plotting combined lemur, build it from the three subsets first
    from src.utils import add_lemur_combined
    metrics         = add_lemur_combined(metrics)
    dataset_metrics = metrics[metrics['dataset'] == dataset].copy()
    codecs          = dataset_metrics['codec'].unique()
    colours = {'encodec': '#1f77b4', 'dac': '#ff7f0e', 'snac': '#2ca02c'}

    # DAC x-axis strategy:
    #   nominal   (current): n_q * (44100/512) * 10 / 1000 -- matches paper, no per-file variation
    #   filesize  (future):  use 'filesize_kbps' column from measure_dac_bitrates.py
    # Switch by changing DAC_XAXIS below.
    DAC_XAXIS = 'nominal'   # change to 'filesize' after empirical .dac run

    # DAC architecture constants for nominal formula (Kumar et al., 2024)
    _DAC_FRAME_RATE  = 44100 / 512
    _DAC_BITS_PER_CB = 10

    dataset_metrics = dataset_metrics.copy()
    dataset_metrics['nominal_kbps'] = (
        dataset_metrics['bitrate'] * _DAC_FRAME_RATE * _DAC_BITS_PER_CB / 1000
    )
    dataset_metrics['plot_bitrate'] = dataset_metrics['bitrate']

    # sharey='row': same y-axis scale across each row for direct comparison
    # sharex='row': same x-axis (bitrate) range across each row
    # Row 0: SI-SNR (whole vs call) -- Row 1: STFT Distance (whole vs call)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey='row', sharex='row')

    metric_pairs = [
        ('si_snr_whole',    'SI-SNR (dB)',        'Whole-recording SI-SNR',        axes[0, 0], 'si_snr'),
        ('si_snr_call_mean','SI-SNR (dB)',        'Call-region SI-SNR',            axes[0, 1], 'si_snr'),
        ('stft_whole',      'STFT Distance (dB)', 'Whole-recording STFT Distance', axes[1, 0], 'stft'),
        ('stft_call_mean',  'STFT Distance (dB)', 'Call-region STFT Distance',     axes[1, 1], 'stft'),
    ]

    for metric_col, ylabel, subplot_title, ax, baseline_key in metric_pairs:
        for codec in codecs:
            colour     = colours.get(codec, 'grey')
            codec_data = dataset_metrics[dataset_metrics['codec'] == codec].copy()

            if codec == 'dac':
                # X position: nominal or filesize kbps depending on DAC_XAXIS.
                # nominal  -- computed from architecture constants, consistent with paper
                # filesize -- from saved .dac file sizes via measure_dac_bitrates.py
                x_col = ('filesize_kbps'
                         if DAC_XAXIS == 'filesize' and 'filesize_kbps' in codec_data.columns
                         else 'nominal_kbps')
                grouped  = (codec_data
                            .groupby('bitrate')
                            .agg(x=(x_col, 'mean'), y=(metric_col, 'mean'))
                            .reset_index()
                            .sort_values('x'))
                plot_x, plot_y = grouped['x'], grouped['y']
            else:
                grouped  = (codec_data
                            .groupby('plot_bitrate')[metric_col]
                            .mean()
                            .reset_index()
                            .sort_values('plot_bitrate'))
                plot_x, plot_y = grouped['plot_bitrate'], grouped[metric_col]

            # Bioacoustic results — solid line
            ax.plot(plot_x, plot_y,
                    marker='o', label=f'{codec.upper()} (bioacoustic)',
                    color=colour, linewidth=2, markersize=6)

            # Paper baselines — dashed line with hollow markers
            baselines = PAPER_BASELINES.get(codec, {}).get(baseline_key, [])
            if baselines:
                bx     = [b[0] for b in baselines]
                by     = [b[1] for b in baselines]
                domain = 'speech/music' if codec == 'encodec' else 'speech/music/env'
                ax.plot(bx, by,
                        marker='o', linestyle='--',
                        label=f'{codec.upper()} (paper, {domain})',
                        color=colour, linewidth=1.5, markersize=6,
                        markerfacecolor='white', alpha=0.7)

        ax.set_xlabel('Bitrate (kbps)')
        ax.set_ylabel(ylabel)
        ax.set_title(subplot_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{title if title else f'Metrics vs Bitrate: {dataset}'}\n"
        f"Dashed lines: paper baselines on speech/music/env sounds",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(figs_dir / f'{dataset}_metrics_vs_bitrate.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_metrics_vs_bitrate_whole(
    metrics:  pd.DataFrame,
    figs_dir: Path,
    title:    str  = None,
    sharey:   bool = True,
) -> None:
    # from src.utils import add_lemur_combined
    plt.rcParams['font.family'] = 'Arial'
    # metrics = add_lemur_combined(metrics)

    DATASET_ORDER = [
        ('anuraset',           'AnuraSet'),
        ('northeastern_birds', 'Northeastern US Soundscapes'),
        ('lemur',              'Black-and-White Ruffed Lemur'),
    ]
    # metric_col now refers to agg csv columns
    METRIC_ROWS = [
        ('si_snr_mean', 'SI-SNR (dB)',   'si_snr'),
        ('stft_mean',   'STFT distance', 'stft'),
    ]
    PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    colours = {
        'encodec': '#1f77b4',
        'dac'    : '#ff7f0e',
        'mp3'    : '#2ca02c',
        'opus'   : '#9467bd',
    }

    fig, axes = plt.subplots(
        2, 3,
        figsize=(22, 16),
        sharey='row' if sharey else False,
        sharex=False,
    )

    legend_handles = []
    legend_labels  = []

    for row_idx, (metric_col, ylabel, published_key) in enumerate(METRIC_ROWS):
        for col_idx, (dataset_key, dataset_label) in enumerate(DATASET_ORDER):
            ax = axes[row_idx, col_idx]
            dataset_data = metrics[metrics['dataset'] == dataset_key].copy()
            if dataset_data.empty:
                ax.set_visible(False)
                continue

            codecs = sorted(dataset_data['codec'].unique())
            for codec in codecs:
                colour     = colours.get(codec, 'grey')
                codec_data = dataset_data[
                    dataset_data['codec'] == codec
                ].sort_values('kbps')

                # kbps already computed in agg csv for all codecs
                plot_x = codec_data['kbps']
                plot_y = codec_data[metric_col]

                line, = ax.plot(
                    plot_x, plot_y,
                    marker='o', linewidth=2, markersize=8,
                    color=colour,
                    label=codec.upper(),
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(codec.upper())

                if codec in ('encodec', 'dac', 'opus'):
                    published = PAPER_BASELINES.get(codec, {}).get(published_key, [])
                    if published:
                        bx = [b[0] for b in published]
                        by = [b[1] for b in published]
                        pub_line, = ax.plot(
                            bx, by,
                            marker='o', linestyle='--',
                            linewidth=1.5, markersize=7,
                            color=colour, alpha=0.7,
                            markerfacecolor='white',
                            label=f'{codec.upper()} (speech/music/env)',
                        )
                        if row_idx == 0 and col_idx == 0:
                            legend_handles.append(pub_line)
                            legend_labels.append(
                                f'{codec.upper()} (speech/music/env)')

            panel_label = PANEL_LABELS[row_idx * 3 + col_idx]
            ax.text(0.02, 0.98, panel_label,
                    transform=ax.transAxes,
                    fontsize=24, va='top', ha='left')
            if row_idx == 0:
                ax.set_title(dataset_label, fontsize=24, pad=10)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=24)
            if row_idx == len(METRIC_ROWS) - 1:
                ax.set_xlabel('Bitrate (kbps)', fontsize=24)
            ax.tick_params(labelsize=24, direction='in')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
            ax.margins(y=0.15)

            if row_idx == 0 and col_idx == 1:
                ax.legend(handles=legend_handles, labels=legend_labels,
                          fontsize=22, frameon=False, loc='best')

    plt.tight_layout()
    out = figs_dir / 'metrics_vs_bitrate_whole.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out.name}")

# def plot_metrics_vs_bitrate_whole(
#     metrics:  pd.DataFrame,
#     figs_dir: Path,
#     title:    str  = None,
#     sharey:   bool = True,
# ) -> None:
#     from src.utils import add_lemur_combined
#     plt.rcParams['font.family'] = 'Arial'
#     metrics = add_lemur_combined(metrics)
#     DATASET_ORDER = [
#         ('anuraset',            'AnuraSet'),
#         ('northeastern_birds',  'Northeastern US Soundscapes'),
#         ('lemur',               'Black-and-White Ruffed Lemur'),
#     ]
#     METRIC_ROWS = [
#         ('si_snr_whole', 'SI-SNR (dB)',   'si_snr'),
#         ('stft_whole',   'STFT distance', 'stft'),
#     ]
#     PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
#     colours = {
#         'encodec': '#1f77b4',
#         'dac':     '#ff7f0e',
#         'mp3':     '#2ca02c',
#         'opus':    '#9467bd',
#     }
#     DAC_FRAME_RATE  = 44100 / 512
#     DAC_BITS_PER_CB = 10
#     metrics = metrics.copy()
#     metrics['nominal_kbps'] = (
#         metrics['bitrate'] * DAC_FRAME_RATE * DAC_BITS_PER_CB / 1000
#     )
#     fig, axes = plt.subplots(
#         2, 3,
#         figsize=(22, 16),
#         sharey='row' if sharey else False,
#         sharex=False,
#     )
#     legend_handles = []
#     legend_labels  = []
#     for row_idx, (metric_col, ylabel, published_key) in enumerate(METRIC_ROWS):
#         for col_idx, (dataset_key, dataset_label) in enumerate(DATASET_ORDER):
#             ax = axes[row_idx, col_idx]
#             dataset_data = metrics[metrics['dataset'] == dataset_key].copy()
#             if dataset_data.empty:
#                 ax.set_visible(False)
#                 continue
#             codecs = sorted(dataset_data['codec'].unique())
#             for codec in codecs:
#                 colour     = colours.get(codec, 'grey')
#                 codec_data = dataset_data[dataset_data['codec'] == codec].copy()
#                 if codec == 'dac':
#                     grouped = (
#                         codec_data
#                         .groupby('bitrate')
#                         .agg(x=('nominal_kbps', 'mean'),
#                              y=(metric_col, 'mean'))
#                         .reset_index()
#                         .sort_values('x')
#                     )
#                     plot_x = grouped['x']
#                     plot_y = grouped['y']
#                 else:
#                     grouped = (
#                         codec_data
#                         .groupby('bitrate')[metric_col]
#                         .mean()
#                         .reset_index()
#                         .sort_values('bitrate')
#                     )
#                     plot_x = grouped['bitrate']
#                     plot_y = grouped[metric_col]
#                 line, = ax.plot(
#                     plot_x, plot_y,
#                     marker='o', linewidth=2, markersize=8,
#                     color=colour,
#                     label=f'{codec.upper()}',
#                 )
#                 if row_idx == 0 and col_idx == 0:
#                     legend_handles.append(line)
#                     legend_labels.append(codec.upper())

#                 if codec in ('encodec', 'dac', 'opus'):
#                     published = PAPER_BASELINES.get(codec, {}).get(published_key, [])
#                     if published:
#                         bx = [b[0] for b in published]
#                         by = [b[1] for b in published]
#                         pub_line, = ax.plot(
#                             bx, by,
#                             marker='o', linestyle='--',
#                             linewidth=1.5, markersize=7,
#                             color=colour, alpha=0.7,
#                             markerfacecolor='white',
#                             label=f'{codec.upper()} (speech/music/env)',
#                         )
#                         if row_idx == 0 and col_idx == 0:
#                             legend_handles.append(pub_line)
#                             legend_labels.append(f'{codec.upper()} (speech/music/env)')

#             panel_label = PANEL_LABELS[row_idx * 3 + col_idx]
#             ax.text(0.02, 0.98, panel_label,
#                     transform=ax.transAxes,
#                     fontsize=24, va='top', ha='left')
#             if row_idx == 0:
#                 ax.set_title(dataset_label, fontsize=24, pad=10)
#             if col_idx == 0:
#                 ax.set_ylabel(ylabel, fontsize=24)
#             if row_idx == len(METRIC_ROWS) - 1:
#                 ax.set_xlabel('Bitrate (kbps)', fontsize=24)
#             ax.tick_params(labelsize=24, direction='in')
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.grid(True, alpha=0.3)
#             ax.margins(y=0.15)

#             # Legend in subplot (b) — row 0, col 1
#             if row_idx == 0 and col_idx == 1:
#                 ax.legend(handles=legend_handles, labels=legend_labels,
#                           fontsize=22, frameon=False, loc='best')

#     plt.tight_layout()
#     out = figs_dir / 'metrics_vs_bitrate_whole.png'
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {out.name}")

# def plot_metrics_vs_bitrate_whole(
#     metrics:  pd.DataFrame,
#     figs_dir: Path,
#     title:    str  = None,
#     sharey:   bool = True,
# ) -> None:
#     """
#     2x3 line plot of whole-recording SI-SNR and STFT distance vs bitrate,
#     one column per dataset (AnuraSet, Northeastern US Soundscapes, Lemur).
#     Each row shares the same y-axis for direct comparison across datasets.
#     Paper baselines shown as dashed lines for reference.
#     Row 0: SI-SNR (dB)
#     Row 1: STFT Distance
#     Columns: AnuraSet | Northeastern US Soundscapes | Lemur (combined)
#     MEE/BES format: no internal title, no bold, Arial font,
#     inward ticks, no top/right spines, panel labels (a)-(f).
#     """
#     from src.utils import add_lemur_combined

#     plt.rcParams['font.family'] = 'Arial'

#     metrics = add_lemur_combined(metrics)

#     DATASET_ORDER = [
#         ('anuraset',            'AnuraSet'),
#         ('northeastern_birds',  'Northeastern US Soundscapes'),
#         ('lemur',               'Black-and-White Ruffed Lemur'),
#     ]
#     METRIC_ROWS = [
#         ('si_snr_whole', 'SI-SNR (dB)',   'si_snr'),
#         ('stft_whole',   'STFT distance', 'stft'),
#     ]
#     PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

#     colours = {
#         'encodec': '#1f77b4',  # blue
#         'dac':     '#ff7f0e',  # orange
#         'mp3':     '#2ca02c',  # green
#         'opus':    '#9467bd',  # purple
#     }

#     DAC_FRAME_RATE  = 44100 / 512
#     DAC_BITS_PER_CB = 10

#     metrics = metrics.copy()
#     metrics['nominal_kbps'] = (
#         metrics['bitrate'] * DAC_FRAME_RATE * DAC_BITS_PER_CB / 1000
#     )

#     fig, axes = plt.subplots(
#         2, 3,
#         figsize=(22, 16),
#         sharey='row' if sharey else False,
#         sharex=False,
#     )

#     for row_idx, (metric_col, ylabel, published_key) in enumerate(METRIC_ROWS):
#         for col_idx, (dataset_key, dataset_label) in enumerate(DATASET_ORDER):
#             ax = axes[row_idx, col_idx]

#             dataset_data = metrics[metrics['dataset'] == dataset_key].copy()
#             if dataset_data.empty:
#                 ax.set_visible(False)
#                 continue

#             codecs = sorted(dataset_data['codec'].unique())

#             for codec in codecs:
#                 colour     = colours.get(codec, 'grey')
#                 codec_data = dataset_data[dataset_data['codec'] == codec].copy()

#                 if codec == 'dac':
#                     grouped = (
#                         codec_data
#                         .groupby('bitrate')
#                         .agg(x=('nominal_kbps', 'mean'),
#                              y=(metric_col, 'mean'))
#                         .reset_index()
#                         .sort_values('x')
#                     )
#                     plot_x = grouped['x']
#                     plot_y = grouped['y']
#                 else:
#                     grouped = (
#                         codec_data
#                         .groupby('bitrate')[metric_col]
#                         .mean()
#                         .reset_index()
#                         .sort_values('bitrate')
#                     )
#                     plot_x = grouped['bitrate']
#                     plot_y = grouped[metric_col]

#                 ax.plot(
#                     plot_x, plot_y,
#                     marker='o', linewidth=2, markersize=8,
#                     color=colour,
#                     label=f'{codec.upper()}',
#                 )

#                 # Published performances — encodec, dac, opus
#                 if codec in ('encodec', 'dac', 'opus'):
#                     published = PAPER_BASELINES.get(codec, {}).get(published_key, [])
#                     if published:
#                         bx = [b[0] for b in published]
#                         by = [b[1] for b in published]
#                         ax.plot(
#                             bx, by,
#                             marker='o', linestyle='--',
#                             linewidth=1.5, markersize=7,
#                             color=colour, alpha=0.7,
#                             markerfacecolor='white',
#                             label=f'{codec.upper()} (speech/music/env)',
#                         )

#             # Panel label (a)-(f) upper left
#             panel_label = PANEL_LABELS[row_idx * 3 + col_idx]
#             ax.text(0.02, 0.98, panel_label,
#                     transform=ax.transAxes,
#                     fontsize=20, va='top', ha='left')

#             if row_idx == 0:
#                 ax.set_title(dataset_label, fontsize=22, pad=10)

#             if col_idx == 0:
#                 ax.set_ylabel(ylabel, fontsize=22)

#             if row_idx == len(METRIC_ROWS) - 1:
#                 ax.set_xlabel('Bitrate (kbps)', fontsize=22)

#             ax.tick_params(labelsize=18, direction='in')
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.grid(True, alpha=0.3)
#             ax.legend(fontsize=16, frameon=False)
#             ax.margins(y=0.15)

#     plt.tight_layout()
#     out = figs_dir / 'metrics_vs_bitrate_whole.png'
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {out.name}")


# FIGURE 2: THREE-PANEL SPECTROGRAM

def plot_three_panel(original:      np.ndarray,
                     reconstructed: np.ndarray,
                     sr:            int,
                     annotations:   pd.DataFrame = None,
                     start_sec:     float = 0,
                     end_sec:       float = 10,
                     max_freq:      int   = None,
                     n_fft:         int   = 2048,
                     hop_length:    int   = 512,
                     title:         str   = None,
                     save_path:     str   = None) -> None:
    """
    Three-panel spectrogram: original, reconstructed, residual.
    Residual is reconstructed minus original — red = signal lost,
    blue = artefact introduced.
    Bounding boxes overlaid in white — for datasets without frequency
    bounds boxes span full spectrogram height showing time boundaries only.
    """
    if max_freq is None:
        max_freq = sr // 2

    orig_slice  = original[int(start_sec*sr):int(end_sec*sr)].astype(np.float32)
    recon_slice = reconstructed[int(start_sec*sr):int(end_sec*sr)].astype(np.float32)
    min_len     = min(len(orig_slice), len(recon_slice))
    orig_slice  = orig_slice[:min_len]
    recon_slice = recon_slice[:min_len]

    D_orig  = _get_db(orig_slice, n_fft, hop_length)
    D_recon = _get_db(recon_slice, n_fft, hop_length)
    D_resid = D_recon - D_orig

    vmax = np.percentile(D_orig, 99)
    vmin = vmax - 40.0 # Set clean bounds dynamic to peak power threshold

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for ax, data, cmap, label, vrange in [
        (axes[0], D_orig,  'magma',  'Original',                        (vmin, vmax)),
        (axes[1], D_recon, 'magma',  'Reconstructed',                   (vmin, vmax)),
        (axes[2], D_resid, 'RdBu_r', 'Residual (Reconstructed - Original)', (-20, 20)),
    ]:
        img = librosa.display.specshow(data, sr=sr, hop_length=hop_length,
                                        x_axis='time', y_axis='hz',
                                        ax=ax, cmap=cmap,
                                        vmin=vrange[0], vmax=vrange[1])
        ax.set_title(label)
        ax.set_ylim(0, max_freq)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    if annotations is not None:
        for ax in axes:
            _overlay_boxes(ax, annotations, start_sec, end_sec, max_freq, sr)

    plt.suptitle(title if title else '', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# FIGURE 3: CALL-LEVEL ZOOM

def find_most_degraded_call(original:      np.ndarray,
                             reconstructed: np.ndarray,
                             annotations:   pd.DataFrame,
                             sr:            int) -> tuple:
    """
    Find the annotation with the worst SI-SNR — the call most
    degraded by compression. Skips calls shorter than 100ms and
    calls that extend beyond the audio length.
    """
    from src.utils import si_snr

    worst_score = float('inf')
    worst_row   = None

    for _, row in annotations.iterrows():
        start = int(row['start_sec'] * sr)
        end   = int(row['end_sec']   * sr)
        if end > len(original) or (end - start) < int(0.1 * sr):
            continue
        score = si_snr(original[start:end], reconstructed[start:end])
        if score < worst_score:
            worst_score = score
            worst_row   = row

    return worst_row, worst_score


def plot_call_zoom(original:      np.ndarray,
                   reconstructed: np.ndarray,
                   sr:            int,
                   call_row:      pd.Series,
                   annotations:   pd.DataFrame = None,
                   padding_sec:   float = 3.0,
                   max_freq:      int   = None,
                   n_fft:         int   = 2048,
                   hop_length:    int   = 512,
                   title:         str   = None,
                   save_path:     str   = None) -> None:
    """
    Three-panel spectrogram zoomed into a single call region.
    Pads the call window by padding_sec on each side for context.
    """
    start_sec = max(0, call_row['start_sec'] - padding_sec)
    end_sec   = call_row['end_sec'] + padding_sec
    plot_three_panel(
        original, reconstructed, sr,
        annotations = annotations,
        start_sec   = start_sec,
        end_sec     = end_sec,
        max_freq    = max_freq,
        n_fft       = n_fft,
        hop_length  = hop_length,
        title       = title,
        save_path   = save_path,
    )


# FIGURE 4: RESIDUAL INSIDE BOUNDING BOX ONLY

def plot_call_residual(original:      np.ndarray,
                       reconstructed: np.ndarray,
                       sr:            int,
                       call_row:      pd.Series,
                       max_freq:      int   = None,
                       n_fft:         int   = 4096,
                       hop_length:    int   = 512,
                       title:         str   = None,
                       save_path:     str   = None) -> None:
    """
    Single-panel residual spectrogram cropped to the exact time-frequency
    bounding box of a specific call. Shows only the frequency range of the
    call, isolating compression artefacts within the biologically relevant
    vocalisation. Red = artefact introduced, blue = signal lost.
    For datasets without frequency bounds (low_freq=0, high_freq=Nyquist),
    shows full frequency range within the call time window.
    """
    if max_freq is None:
        max_freq = sr // 2

    nyquist   = sr // 2
    start     = int(call_row['start_sec'] * sr)
    end       = int(call_row['end_sec']   * sr)
    low_freq  = float(call_row.get('low_freq',  0))
    high_freq = float(call_row.get('high_freq', nyquist))

    # If no frequency bounds — use full spectrogram height
    if low_freq == 0 and high_freq >= nyquist:
        high_freq = max_freq

    # Extract call time slice
    orig_slice  = original[start:end].astype(np.float32)
    recon_slice = reconstructed[start:end].astype(np.float32)
    min_len     = min(len(orig_slice), len(recon_slice))
    orig_slice  = orig_slice[:min_len]
    recon_slice = recon_slice[:min_len]

    # Compute residual spectrogram
    D_orig  = _get_db(orig_slice,  n_fft, hop_length)
    D_recon = _get_db(recon_slice, n_fft, hop_length)
    D_resid = D_recon - D_orig

    # Crop to frequency bounds
    n_bins   = D_resid.shape[0]
    low_bin  = max(0,      int(low_freq  / nyquist * n_bins))
    high_bin = min(n_bins, int(high_freq / nyquist * n_bins))
    D_resid_cropped = D_resid[low_bin:high_bin, :]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    img = librosa.display.specshow(
        D_resid_cropped,
        sr         = sr,
        hop_length = hop_length,
        x_axis     = 'time',
        y_axis     = 'hz',
        ax         = ax,
        cmap       = 'RdBu_r',
        vmin       = -20,
        vmax       = 20,
        fmin       = low_freq,
        fmax       = high_freq,
    )
    ax.set_title('Residual within call bounding box (Reconstructed − Original)')
    ax.set_ylim(low_freq, high_freq)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Add call metadata to title
    label    = call_row.get('label', '')
    duration = call_row['end_sec'] - call_row['start_sec']
    subtitle = (f"Label: {label} | "
                f"Duration: {duration:.2f}s | "
                f"Freq: {low_freq:.0f}–{high_freq:.0f} Hz")
    ax.set_xlabel(subtitle)

    plt.suptitle(title if title else 'Call Residual', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_residual_spectrum(
    dataset:   str,
    metrics:   pd.DataFrame,
    data_dir,
    recon_dir,
    n_fft:     int  = 2048,
    save_path: str  = None,
) -> None:
    """
    Plot mean relative spectral distortion (dB) vs frequency for each
    (codec, bitrate) combination evaluated on a dataset.

    Relative residual = 20 * log10(mean|STFT(recon-orig)| / mean|STFT(orig)|)
    where the mean is taken across all time frames and all files in the
    linear domain before converting to dB.
    """
    import re
    import librosa
    from pathlib import Path as _Path
    from src.utils   import load_audio, resample, peak_normalise
    from src.evaluate import DATASETS

    dataset_metrics = metrics[metrics['dataset'] == dataset].copy()
    if dataset_metrics.empty:
        print(f"No metrics found for dataset '{dataset}'")
        return

    config        = DATASETS[dataset]
    codec_sr_map  = config['codec_sr']
    audio_dir     = _Path(config['audio_dir'])
    audio_subdirs = config.get('audio_subdirs')

    CODEC_COLOURS = {'encodec': '#1f77b4', 'dac': '#ff7f0e'}

    fig, ax = plt.subplots(figsize=(11, 5))

    for codec in sorted(dataset_metrics['codec'].unique()):
        codec_data = dataset_metrics[dataset_metrics['codec'] == codec]
        codec_sr   = codec_sr_map[codec]
        hop        = n_fft // 4
        bitrates   = sorted(codec_data['bitrate'].unique())
        alphas     = np.linspace(0.35, 1.0, len(bitrates))
        colour     = CODEC_COLOURS.get(codec, None)

        orig_sums   = {}   # filename -> (sum_mag, n_frames)
        orig_counts = {}

        for filename in codec_data['filename'].unique():
            if audio_subdirs:
                audio_path = audio_dir / audio_subdirs[0] / filename
            elif dataset == 'anuraset':
                m = re.match(r'^([A-Za-z0-9]+?)_\d{8}_', filename)
                recorder   = m.group(1) if m else ''
                audio_path = audio_dir / recorder / filename
            else:
                audio_path = audio_dir / filename

            if not audio_path.exists():
                continue
            try:
                audio, native_sr = load_audio(str(audio_path))
                audio = peak_normalise(resample(peak_normalise(audio),
                                                 native_sr, codec_sr))
                mag   = np.abs(librosa.stft(audio,
                                             n_fft=n_fft, hop_length=hop))
                orig_sums[filename]   = mag.sum(axis=1)   # sum over frames
                orig_counts[filename] = mag.shape[1]      # n_frames
            except Exception:
                continue

        for bitrate, alpha in zip(bitrates, alphas):
            br_data = codec_data[codec_data['bitrate'] == bitrate]

            sum_resid  = None
            sum_orig   = None
            total_frames = 0

            for _, row in br_data.iterrows():
                filename = row['filename']
                stem     = _Path(filename).stem

                if filename not in orig_sums:
                    continue

                recon_path = (_Path(recon_dir) / dataset / codec /
                              str(bitrate) / (stem + '_reconstructed.wav'))
                if not recon_path.exists():
                    continue

                try:
                    recon, _ = load_audio(str(recon_path))

                    if audio_subdirs:
                        audio_path = audio_dir / audio_subdirs[0] / filename
                    elif dataset == 'anuraset':
                        m = re.match(r'^([A-Za-z0-9]+?)_\d{8}_', filename)
                        recorder   = m.group(1) if m else ''
                        audio_path = audio_dir / recorder / filename
                    else:
                        audio_path = audio_dir / filename

                    audio, native_sr = load_audio(str(audio_path))
                    audio = peak_normalise(resample(peak_normalise(audio),
                                                     native_sr, codec_sr))
                    min_len = min(len(audio), len(recon))
                    residual = recon[:min_len] - audio[:min_len]

                    mag_resid = np.abs(librosa.stft(residual,
                                                     n_fft=n_fft,
                                                     hop_length=hop))
                    n_fr      = mag_resid.shape[1]

                    if sum_resid is None:
                        sum_resid = mag_resid.sum(axis=1)
                        sum_orig  = orig_sums[filename] * (
                            n_fr / orig_counts[filename])
                    else:
                        sum_resid += mag_resid.sum(axis=1)
                        sum_orig  += orig_sums[filename] * (
                            n_fr / orig_counts[filename])
                    total_frames += n_fr
                except Exception:
                    continue

            if sum_resid is None or total_frames == 0:
                continue

            mean_resid   = sum_resid / total_frames
            mean_orig    = sum_orig  / total_frames
            relative_db  = 20 * np.log10(mean_resid / (mean_orig + 1e-8))

            freqs = librosa.fft_frequencies(sr=codec_sr, n_fft=n_fft)

            if codec == 'dac':
                label = f'DAC $N_q$={int(bitrate)}'
            else:
                label = f'EnCodec {bitrate} kbps'

            ax.plot(freqs, relative_db, color=colour,
                    alpha=alpha, label=label, linewidth=1.2)

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.4,
               label='No distortion')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Relative Residual (dB)')
    ax.set_title(f'Spectral Distortion vs Frequency: {dataset}')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# FIGURE 5: SPECIES DEGRADATION WITH SEPARATE QUALITY INDICATORS & LEMUR FIX

def plot_species_degradation(
    metrics:      pd.DataFrame,
    call_metrics: pd.DataFrame,
    codec:        str,
    bitrate,
    metric:       str  = 'si_snr',   # 'si_snr', 'stft_distance', or 'loudness'
    statistical:  bool = False,
    n_label:      int  = 3,          # Number of top/bottom species to label
    data_dir           = None,       # required when metric='loudness'
    recon_dir          = None,       # required when metric='loudness'
    save_path:    str  = None,
) -> None:
    """
    For each dataset, plot one dot per species/variant showing the degradation
    difference between call-region and whole-recording metric values.

    Fixes implemented:
      - Lemur: Filters out 'no-roar', keeps 'roar', skips labelling (not informative).
      - AnuraSet: Keeps H/M/L quality suffixes separate, maps them to distinct
        marker shapes (circle=H, triangle=M, square=L). Labels shown as
        'SPECIES (L)' format.
      - Top-N labelling: labels top N helped + top N hurt species per dataset,
        restricted to statistically significant dots only (p<0.05). Lemur skipped.
    """
    import re
    from pathlib import Path as _Path
    from scipy import stats

    lemur_subsets  = {'lemur_s4a', 'lemur_swift1', 'lemur_swift2'}
    EXCLUDED_LABELS = {'no-roar', 'no_roar', 'noroar'}

    # ── LOUDNESS BRANCH ───────────────────────────────────────────────────────
    if metric == 'loudness':
        if data_dir is None or recon_dir is None:
            raise ValueError("data_dir and recon_dir are required when metric='loudness'")

        from src.utils    import load_audio, resample
        from src.evaluate import DATASETS

        data_dir  = _Path(data_dir)
        recon_dir = _Path(recon_dir)

        calls_raw = call_metrics[
            (call_metrics['codec']   == codec) &
            (call_metrics['bitrate'] == bitrate)
        ][['filename', 'dataset', 'label', 'start_sec', 'end_sec']].copy()

        # Filter out lemur background labels
        calls_raw = calls_raw[
            ~calls_raw['label'].astype(str).str.lower().isin(EXCLUDED_LABELS)
        ]

        calls_raw['dataset_plot'] = calls_raw['dataset'].apply(
            lambda d: 'lemur' if d in lemur_subsets else d)

        loudness_rows = []

        for (dataset_name, filename), file_calls in calls_raw.groupby(['dataset', 'filename']):
            config        = DATASETS[dataset_name]
            codec_sr      = config['codec_sr'][codec]
            audio_dir     = _Path(config['audio_dir'])
            audio_subdirs = config.get('audio_subdirs')
            stem          = _Path(filename).stem

            if audio_subdirs:
                audio_path = audio_dir / audio_subdirs[0] / filename
            elif dataset_name == 'anuraset':
                m = re.match(r'^([A-Za-z0-9]+?)_\d{8}_', filename)
                recorder   = m.group(1) if m else ''
                audio_path = audio_dir / recorder / filename
            else:
                audio_path = audio_dir / filename

            recon_path = (recon_dir / dataset_name / codec /
                          str(bitrate) / (stem + '_reconstructed.wav'))

            if not audio_path.exists() or not recon_path.exists():
                continue

            try:
                audio, native_sr = load_audio(str(audio_path))
                audio = resample(audio, native_sr, codec_sr)
                recon, _         = load_audio(str(recon_path))
                min_len          = min(len(audio), len(recon))
                audio            = audio[:min_len]
                recon            = recon[:min_len]

                dataset_plot = file_calls['dataset_plot'].iloc[0]

                for _, call_row in file_calls.iterrows():
                    start = int(call_row['start_sec'] * codec_sr)
                    end   = int(call_row['end_sec']   * codec_sr)
                    end   = min(end, min_len)
                    if end <= start:
                        continue

                    rms_orig      = float(np.sqrt(np.mean(audio[start:end] ** 2)))
                    rms_recon     = float(np.sqrt(np.mean(recon[start:end] ** 2)))
                    loudness_diff = rms_orig - rms_recon

                    loudness_rows.append({
                        'dataset'    : dataset_plot,
                        'filename'   : filename,
                        'label'      : call_row['label'],
                        'degradation': loudness_diff,
                    })

            except Exception:
                continue

        if not loudness_rows:
            print("No loudness data could be computed.")
            return

        merged = pd.DataFrame(loudness_rows)
        merged = (merged
                  .groupby(['filename', 'dataset', 'label'])['degradation']
                  .mean()
                  .reset_index())

        metric_label = 'RMS Loudness Change (original - reconstructed)'
        direction    = 'positive = codec attenuates calls, negative = codec amplifies'

    # ── SI-SNR / STFT BRANCH ─────────────────────────────────────────────────
    else:
        whole_col = 'si_snr_whole' if metric == 'si_snr' else 'stft_whole'
        call_col  = 'si_snr'       if metric == 'si_snr' else 'stft_dist'

        whole = metrics[
            (metrics['codec']   == codec) &
            (metrics['bitrate'] == bitrate)
        ][['filename', 'dataset', whole_col]].copy()
        whole = whole.rename(columns={whole_col: 'whole'})

        calls_raw = call_metrics[
            (call_metrics['codec']   == codec) &
            (call_metrics['bitrate'] == bitrate)
        ][['filename', 'dataset', 'label', call_col]].copy()

        # Filter out lemur background labels
        calls_raw = calls_raw[
            ~calls_raw['label'].astype(str).str.lower().isin(EXCLUDED_LABELS)
        ]

        calls = (calls_raw
                 .groupby(['filename', 'dataset', 'label'])[call_col]
                 .mean()
                 .reset_index()
                 .rename(columns={call_col: 'call'}))

        whole['dataset'] = whole['dataset'].apply(
            lambda d: 'lemur' if d in lemur_subsets else d)
        calls['dataset'] = calls['dataset'].apply(
            lambda d: 'lemur' if d in lemur_subsets else d)

        merged = calls.merge(whole, on=['filename', 'dataset'], how='inner')
        merged['degradation'] = merged['call'] - merged['whole']

        metric_label = ('SI-SNR (call − whole, dB)'
                        if metric == 'si_snr'
                        else 'STFT distance (call − whole, dB)')
        direction    = ('negative = call region reconstructed less faithfully than background'
                        if metric == 'si_snr'
                        else 'positive = call region more spectrally distorted than background')

    # ── AGGREGATE PER DATASET & SPECIES/VARIANT ──────────────────────────────
    rows = []
    for (dataset, label), group in merged.groupby(['dataset', 'label']):
        # Normalise lemur label
        if dataset == 'lemur':
            label = 'roar'

        values = group['degradation'].dropna().values
        if len(values) == 0:
            continue
        mean = values.mean()

        if statistical:
            if len(values) >= 2:
                _, p_value = stats.ttest_1samp(values, popmean=0)
                alpha = 0.85 if p_value < 0.05 else 0.2
            else:
                p_value = 1.0
                alpha   = 0.2
        else:
            p_value = 0.01
            alpha   = 0.7

        rows.append({'dataset': dataset, 'label': label,
                     'mean': mean, 'alpha': alpha, 'p_value': p_value})

    agg = pd.DataFrame(rows)
    if agg.empty:
        print("No data to plot.")
        return

    # ── PLOT ─────────────────────────────────────────────────────────────────
    datasets = sorted(agg['dataset'].unique())
    x_map    = {d: i for i, d in enumerate(datasets)}

    DATASET_COLOURS = {
        'anuraset':           '#2ca02c',
        'northeastern_birds': '#1f77b4',
        'lemur':              '#ff7f0e',
    }
    QUALITY_MARKERS = {'H': 'o', 'M': '^', 'L': 's', '': 'o'}

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 3), 6))
    rng = np.random.default_rng(42)

    stored_positions = []

    for _, row in agg.iterrows():
        dset   = row['dataset']
        x      = x_map[dset]
        jitter = rng.uniform(-0.18, 0.18)
        colour = DATASET_COLOURS.get(dset, 'grey')

        marker = 'o'
        if dset == 'anuraset' and '_' in str(row['label']):
            suffix = str(row['label']).split('_')[-1]
            marker = QUALITY_MARKERS.get(suffix, 'o')

        ax.scatter(
            x + jitter, row['mean'],
            color=colour, alpha=float(row['alpha']),
            s=65, edgecolors='none', marker=marker,
        )

        stored_positions.append({
            'dataset': dset,
            'x_pos'  : x + jitter,
            'y_pos'  : row['mean'],
            'label'  : row['label'],
            'p_value': row['p_value'],
        })

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.4)

    # ── TOP-N SIGNIFICANT LABELLING ───────────────────────────────────────────
    # Labels top N most helped and top N most hurt species per dataset.
    # Restricted to statistically significant dots (p<0.05).
    # Lemur skipped (single 'roar' label is not informative).
    pos_df = pd.DataFrame(stored_positions)
    texts  = []  # collected for adjust_text()
    if not pos_df.empty and n_label > 0:
        for dset, dset_grp in pos_df.groupby('dataset'):
            if dset == 'lemur':
                continue

            signif_grp = dset_grp[dset_grp['p_value'] < 0.05].copy()
            if signif_grp.empty:
                continue

            sorted_grp = signif_grp.sort_values('y_pos')

            # Bottom N (most hurt) + top N (most helped), deduplicated
            bottom   = sorted_grp.head(n_label)
            top      = sorted_grp.tail(n_label)
            to_label = pd.concat([bottom, top]).drop_duplicates()

            for _, item in to_label.iterrows():
                display_text = str(item['label'])

                # AnuraSet: format as 'SPECIES (L)' instead of 'SPECIES_L'
                if dset == 'anuraset' and '_' in display_text:
                    base, suff   = display_text.rsplit('_', 1)
                    display_text = f"{base} ({suff})"

                t = ax.text(
                    item['x_pos'], item['y_pos'],
                    display_text,
                    fontsize=8, alpha=0.9, va='center', ha='left',
                )
                texts.append(t)

    # Automatically reposition labels to avoid overlaps
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.5),
                expand=(1.2, 1.4),
                force_text=(0.3, 0.5),
            )
        except ImportError:
            pass  # adjustText not installed -- labels stay as-is

    # ── AXES & LABELS ─────────────────────────────────────────────────────────
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        [d.replace('_', '\n').title() for d in datasets], fontsize=10)
    ax.set_ylabel(metric_label)

    title = (f'Per-Species Relative Call-Region Reconstruction Quality: {codec.upper()} @ {bitrate} kbps\n'
             f'({direction})')
    if statistical:
        title += ('\nOpaque = p < 0.05 (significant) | '
                  'Markers: ●=High ▲=Medium ■=Low quality')
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


# FIGURE 6: TIME-FREQUENCY COMPLEXITY SATELLITES

def _compute_aci(audio: np.ndarray, sr: int,
                 n_fft: int = 1024, hop_length: int = 512) -> float:
    """
    Acoustic Complexity Index (Pieretti et al. 2011).
    """
    import librosa
    S    = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    diff = np.abs(np.diff(S, axis=1))
    denom = S[:, :-1].sum(axis=1) + 1e-8
    return float((diff.sum(axis=1) / denom).sum())


def _si_snr_window(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SNR for a short window."""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha  = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise  = est - target
    return float(10 * np.log10(
        (np.dot(target, target) + 1e-8) /
        (np.dot(noise,  noise)  + 1e-8)))


def _stft_dist_window(ref: np.ndarray, est: np.ndarray,
                      sr: int) -> float:
    """STFT distance for a short window across 3 scales."""
    import librosa
    scales = [512, 1024, 2048]
    dists  = []
    for s in scales:
        hop = s // 4
        S_ref = np.abs(librosa.stft(ref, n_fft=s, hop_length=hop))
        S_est = np.abs(librosa.stft(est, n_fft=s, hop_length=hop))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ref = np.log(S_ref + 1e-8)
            log_est = np.log(S_est + 1e-8)
        dists.append(np.mean(np.abs(log_ref - log_est)))
    return float(np.mean(dists))


def plot_complexity_vs_quality_whole(
    metrics:    pd.DataFrame,
    data_dir,
    recon_dir,
    codec:      str,
    bitrate,
    metric:     str = 'si_snr',    # 'si_snr' or 'stft_distance'
    window_sec: int = 60,
    save_path:  str = None,
) -> None:
    """
    Scatter plot of ACI (soundscape complexity) vs per-minute reconstruction
    quality for each dataset.
    """
    import re
    import librosa
    from pathlib import Path as _Path
    from src.utils   import load_audio, resample, peak_normalise
    from src.evaluate import DATASETS

    data_dir  = _Path(data_dir)
    recon_dir = _Path(recon_dir)

    MARKERS = {
        'anuraset':           'o',
        'northeastern_birds': '^',
        'lemur':              's',
    }
    COLOURS = {
        'anuraset':           '#2ca02c',
        'northeastern_birds': '#1f77b4',
        'lemur':              '#ff7f0e',
    }
    LEMUR_SUBSETS = {'lemur_s4a', 'lemur_swift1', 'lemur_swift2'}

    br_metrics = metrics[
        (metrics['codec']   == codec) &
        (metrics['bitrate'] == bitrate)
    ].copy()
    br_metrics['dataset_plot'] = br_metrics['dataset'].apply(
        lambda d: 'lemur' if d in LEMUR_SUBSETS else d)

    metric_fn = _si_snr_window if metric == 'si_snr' else _stft_dist_window
    ylabel    = 'SI-SNR (dB)' if metric == 'si_snr' else 'STFT Distance (dB)'

    records = []

    for dataset_plot, group in br_metrics.groupby('dataset_plot'):
        dataset_name = group['dataset'].iloc[0]
        config       = DATASETS[dataset_name]
        codec_sr     = config['codec_sr'][codec]
        audio_dir    = _Path(config['audio_dir'])
        audio_subdirs = config.get('audio_subdirs')
        win_samples  = window_sec * codec_sr

        for _, row in group.iterrows():
            filename = row['filename']

            if audio_subdirs:
                audio_path = audio_dir / audio_subdirs[0] / filename
            elif dataset_name == 'anuraset':
                m = re.match(r'^([A-Za-z0-9]+?)_\d{8}_', filename)
                recorder   = m.group(1) if m else ''
                audio_path = audio_dir / recorder / filename
            else:
                audio_path = audio_dir / filename

            recon_path = (recon_dir / dataset_name / codec /
                          str(bitrate) / (_Path(filename).stem + '_reconstructed.wav'))

            if not audio_path.exists() or not recon_path.exists():
                continue

            try:
                audio, native_sr = load_audio(str(audio_path))
                audio = peak_normalise(resample(
                    peak_normalise(audio), native_sr, codec_sr))
                recon, _ = load_audio(str(recon_path))
                min_len  = min(len(audio), len(recon))
                audio    = audio[:min_len]
                recon    = recon[:min_len]

                n_windows = max(1, min_len // win_samples)
                for i in range(n_windows):
                    s = i * win_samples
                    e = s + win_samples
                    if e > min_len:
                        break
                    a_win = audio[s:e]
                    r_win = recon[s:e]

                    aci     = _compute_aci(a_win, codec_sr)
                    quality = (metric_fn(a_win, r_win) if metric == 'si_snr'
                               else metric_fn(a_win, r_win, codec_sr))
                    records.append((dataset_plot, aci, quality))

            except Exception:
                continue

    if not records:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    df = pd.DataFrame(records, columns=['dataset', 'aci', 'quality'])

    for dset, grp in df.groupby('dataset'):
        ax.scatter(
            grp['aci'], grp['quality'],
            color=COLOURS.get(dset, 'grey'),
            marker=MARKERS.get(dset, 'o'),
            alpha=0.5, s=30, edgecolors='none',
            label=dset.replace('_', ' ').title(),
        )

    ax.set_xlabel('ACI (Acoustic Complexity Index)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Soundscape Complexity vs Reconstruction Quality\n{codec.upper()} @ {bitrate} kbps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_complexity_vs_quality_dataset(
    metrics:    pd.DataFrame,
    data_dir,
    recon_dir,
    codec:      str,
    bitrate,
    dataset:    str,
    metric:     str  = 'si_snr',
    statistical: bool = False,
    window_sec: int  = 60,
    save_path:  str  = None,
) -> None:
    """
    Scatter plot of ACI vs per-minute reconstruction quality for a single dataset.
    """
    import re
    from scipy import stats
    from pathlib import Path as _Path
    from src.utils   import load_audio, resample, peak_normalise
    from src.evaluate import DATASETS

    data_dir  = _Path(data_dir)
    recon_dir = _Path(recon_dir)

    LEMUR_SUBSETS = {'lemur_s4a', 'lemur_swift1', 'lemur_swift2'}
    COLOURS = {'anuraset': '#2ca02c', 'northeastern_birds': '#1f77b4', 'lemur': '#ff7f0e'}
    MARKERS = {'anuraset': 'o', 'northeastern_birds': '^', 'lemur': 's'}

    dataset_plot = 'lemur' if dataset in LEMUR_SUBSETS else dataset

    if dataset_plot == 'lemur':
        br_metrics = metrics[
            (metrics['codec']   == codec) &
            (metrics['bitrate'] == bitrate) &
            (metrics['dataset'].isin(LEMUR_SUBSETS))
        ].copy()
    else:
        br_metrics = metrics[
            (metrics['codec']   == codec) &
            (metrics['bitrate'] == bitrate) &
            (metrics['dataset'] == dataset)
        ].copy()

    if br_metrics.empty:
        print(f"No metrics found for dataset '{dataset}'")
        return

    metric_fn = _si_snr_window if metric == 'si_snr' else _stft_dist_window
    ylabel    = 'SI-SNR (dB)' if metric == 'si_snr' else 'STFT Distance (dB)'

    records = []

    for _, row in br_metrics.iterrows():
        dataset_name  = row['dataset']
        filename      = row['filename']
        config        = DATASETS[dataset_name]
        codec_sr      = config['codec_sr'][codec]
        audio_dir     = _Path(config['audio_dir'])
        audio_subdirs = config.get('audio_subdirs')
        win_samples   = window_sec * codec_sr

        if audio_subdirs:
            audio_path = audio_dir / audio_subdirs[0] / filename
        elif dataset_name == 'anuraset':
            m = re.match(r'^([A-Za-z0-9]+?)_\d{8}_', filename)
            recorder   = m.group(1) if m else ''
            audio_path = audio_dir / recorder / filename
        else:
            audio_path = audio_dir / filename

        recon_path = (recon_dir / dataset_name / codec /
                      str(bitrate) / (_Path(filename).stem + '_reconstructed.wav'))

        if not audio_path.exists() or not recon_path.exists():
            continue

        try:
            audio, native_sr = load_audio(str(audio_path))
            audio = peak_normalise(resample(peak_normalise(audio), native_sr, codec_sr))
            recon, _ = load_audio(str(recon_path))
            min_len   = min(len(audio), len(recon))
            audio     = audio[:min_len]
            recon     = recon[:min_len]

            n_windows = max(1, min_len // win_samples)
            for i in range(n_windows):
                s = i * win_samples
                e = s + win_samples
                if e > min_len:
                    break
                a_win = audio[s:e]
                r_win = recon[s:e]
                aci     = _compute_aci(a_win, codec_sr)
                quality = (metric_fn(a_win, r_win) if metric == 'si_snr'
                           else metric_fn(a_win, r_win, codec_sr))
                records.append((aci, quality))
        except Exception:
            continue

    if not records:
        print("No data to plot.")
        return

    aci_vals     = np.array([r[0] for r in records])
    quality_vals = np.array([r[1] for r in records])

    if statistical and len(records) >= 3:
        r_val, p_val = stats.pearsonr(aci_vals, quality_vals)
        significant  = p_val < 0.05
        alpha        = 0.6 if significant else 0.15
    else:
        r_val, p_val  = None, None
        significant   = None
        alpha         = 0.5

    colour = COLOURS.get(dataset_plot, 'grey')
    marker = MARKERS.get(dataset_plot, 'o')

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(aci_vals, quality_vals, color=colour, marker=marker, alpha=alpha, s=30, edgecolors='none')

    if statistical and significant and len(records) >= 3:
        slope, intercept, _, _, se = stats.linregress(aci_vals, quality_vals)
        x_line = np.linspace(aci_vals.min(), aci_vals.max(), 200)
        y_line = slope * x_line + intercept

        n     = len(aci_vals)
        x_bar = aci_vals.mean()
        t_crit = stats.t.ppf(0.975, df=n - 2)
        se_line = se * np.sqrt(1/n + (x_line - x_bar)**2 / ((aci_vals - x_bar)**2).sum())
        ci      = t_crit * se_line

        ax.plot(x_line, y_line, color=colour, linewidth=1.5, alpha=0.8)
        ax.fill_between(x_line, y_line - ci, y_line + ci, color=colour, alpha=0.15, label='95% CI')

        p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
        ax.annotate(f'r = {r_val:.3f}, {p_str}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('ACI (Acoustic Complexity Index)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _bandpass_filter(audio: np.ndarray, sr: int,
                     low_freq: float, high_freq: float) -> np.ndarray:
    """Apply a Butterworth bandpass filter to audio."""
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2.0
    low     = max(low_freq,  1.0)
    high    = min(high_freq, nyquist - 1.0)

    if low >= high:
        return audio
    sos = butter(4, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def plot_species_frequency_band(
    call_metrics: pd.DataFrame,
    codec:        str,
    bitrate,
    metric:       str  = 'si_snr',
    statistical:  bool = False,
    data_dir            = None,
    recon_dir           = None,
    save_path:    str  = None,
) -> None:
    """Plot absolute within-band metric value per species range."""
    import re
    from pathlib import Path as _Path
    from scipy import stats
    from src.utils   import load_audio, resample
    from src.evaluate import DATASETS

    if data_dir is None or recon_dir is None:
        raise ValueError("data_dir and recon_dir are required")

    data_dir  = _Path(data_dir)
    recon_dir = _Path(recon_dir)

    LEMUR_SUBSETS    = {'lemur_s4a', 'lemur_swift1', 'lemur_swift2'}
    SKIP_DATASETS    = {'anuraset'}
    DATASET_COLOURS  = {'northeastern_birds': '#1f77b4', 'lemur': '#ff7f0e'}

    calls_raw = call_metrics[
        (call_metrics['codec']   == codec) &
        (call_metrics['bitrate'] == bitrate) &
        (~call_metrics['dataset'].isin(SKIP_DATASETS))
    ].copy()

    if calls_raw.empty:
        return

    calls_raw['dataset_plot'] = calls_raw['dataset'].apply(lambda d: 'lemur' if d in LEMUR_SUBSETS else d)
    fullband_col = 'si_snr' if metric == 'si_snr' else 'stft_dist'
    result_rows = []

    for (dataset_name, filename), file_calls in calls_raw.groupby(['dataset', 'filename']):
        config        = DATASETS[dataset_name]
        codec_sr      = config['codec_sr'][codec]
        audio_dir     = _Path(config['audio_dir'])
        audio_subdirs = config.get('audio_subdirs')
        stem          = _Path(filename).stem
        dataset_plot  = file_calls['dataset_plot'].iloc[0]

        audio_path = audio_dir / audio_subdirs[0] / filename if audio_subdirs else audio_dir / filename
        recon_path = recon_dir / dataset_name / codec / str(bitrate) / (stem + '_reconstructed.wav')

        if not audio_path.exists() or not recon_path.exists():
            continue

        try:
            audio, native_sr = load_audio(str(audio_path))
            audio = resample(audio, native_sr, codec_sr)
            recon, _  = load_audio(str(recon_path))
            min_len   = min(len(audio), len(recon))
            audio     = audio[:min_len]
            recon     = recon[:min_len]

            for _, call_row in file_calls.iterrows():
                low_freq  = float(call_row.get('low_freq',  0))
                high_freq = float(call_row.get('high_freq', codec_sr / 2))
                nyquist   = codec_sr / 2

                if low_freq <= 0 and high_freq >= nyquist * 0.95:
                    continue

                start = int(call_row['start_sec'] * codec_sr)
                end   = min(int(call_row['end_sec'] * codec_sr), min_len)
                if end <= start:
                    continue

                orig_band  = _bandpass_filter(audio[start:end],  codec_sr, low_freq, high_freq)
                recon_band = _bandpass_filter(recon[start:end], codec_sr, low_freq, high_freq)

                band_val = _si_snr_window(orig_band, recon_band) if metric == 'si_snr' else _stft_dist_window(orig_band, recon_band, codec_sr)
                result_rows.append({'dataset': dataset_plot, 'filename': filename, 'label': call_row['label'], 'band_metric': band_val, 'full_metric': float(call_row[fullband_col])})
        except Exception:
            continue

    if not result_rows:
        return

    df = pd.DataFrame(result_rows).groupby(['filename', 'dataset', 'label']).mean().reset_index()
    rows = []
    for (dataset, label), group in df.groupby(['dataset', 'label']):
        band_vals = group['band_metric'].dropna().values
        full_vals = group['full_metric'].dropna().values
        if len(band_vals) == 0: continue

        if statistical and len(band_vals) >= 2 and len(full_vals) >= 2:
            _, p_value = stats.ttest_rel(band_vals, full_vals)
            alpha = 0.85 if p_value < 0.05 else 0.2
        else:
            p_value, alpha = 1.0, 0.7

        rows.append({'dataset': dataset, 'label': label, 'mean': band_vals.mean(), 'alpha': alpha, 'p_value': p_value})

    agg = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(6, len(sorted(agg['dataset'].unique())) * 2.5), 6))
    rng = np.random.default_rng(42)

    for _, row in agg.iterrows():
        x = sorted(agg['dataset'].unique()).index(row['dataset'])
        ax.scatter(x + rng.uniform(-0.15, 0.15), row['mean'], color=DATASET_COLOURS.get(row['dataset'], 'grey'), alpha=float(row['alpha']), s=60)

    ax.set_xticks(range(len(sorted(agg['dataset'].unique()))))
    ax.set_xticklabels([d.replace('_', '\n').title() for d in sorted(agg['dataset'].unique())])
    ax.set_ylabel(f'Within-Band {"SI-SNR (dB)" if metric == "si_snr" else "STFT Distance (dB)"}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


# FIGURE 7: FREQUENCY BANDWISE ERROR DISTRIBUTION BAR CHART

def plot_frequency_band_distortion(
    metrics:    pd.DataFrame,
    data_dir,
    recon_dir,
    codec_bitrates: dict = None,
    n_fft:      int  = 2048,
    bands:      list = None,
    save_path:  str  = None,
) -> None:
    """Grouped bar chart showing mean relative spectral distortion per frequency band."""
    import re
    import librosa
    from pathlib import Path as _Path
    from src.utils   import load_audio, resample, peak_normalise
    from src.evaluate import DATASETS

    data_dir  = _Path(data_dir)
    recon_dir = _Path(recon_dir)

    if codec_bitrates is None: codec_bitrates = {'encodec': 6.0, 'dac': 6}
    if bands is None:
        bands = [(0, 2000, '0-2k'), (2000, 4000, '2-4k'), (4000, 8000, '4-8k'), (8000, 12000, '8-12k')]

    CODEC_COLOURS = {'encodec': '#1f77b4', 'dac': '#ff7f0e'}
    LEMUR_SUBSETS = {'lemur_s4a', 'lemur_swift1', 'lemur_swift2'}

    metrics = metrics.copy()
    metrics['dataset_plot'] = metrics['dataset'].apply(lambda d: 'lemur' if d in LEMUR_SUBSETS else d)

    datasets_plot = sorted(metrics['dataset_plot'].unique())
    n_datasets    = len(datasets_plot)
    n_bands       = len(bands)
    hop           = n_fft // 4

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5), sharey=True)
    if n_datasets == 1: axes = [axes]

    bar_width = 0.35
    x         = np.arange(n_bands)

    for ax, dataset_plot in zip(axes, datasets_plot):
        dataset_names = [d for d in metrics['dataset'].unique() if d in LEMUR_SUBSETS] if dataset_plot == 'lemur' else [dataset_plot]

        for codec_idx, (codec, bitrate) in enumerate(codec_bitrates.items()):
            colour = CODEC_COLOURS.get(codec, 'grey')
            band_residuals = [[] for _ in range(n_bands)]

            for dataset_name in dataset_names:
                if dataset_name not in DATASETS: continue
                config        = DATASETS[dataset_name]
                codec_sr      = config['codec_sr'][codec]
                audio_dir     = _Path(config['audio_dir'])
                audio_subdirs = config.get('audio_subdirs')

                subset = metrics[(metrics['dataset']  == dataset_name) & (metrics['codec'] == codec) & (metrics['bitrate'] == bitrate)]
                if subset.empty: continue
                freqs = librosa.fft_frequencies(sr=codec_sr, n_fft=n_fft)

                for _, row in subset.iterrows():
                    filename = row['filename']
                    audio_path = audio_dir / audio_subdirs[0] / filename if audio_subdirs else audio_dir / filename
                    recon_path = recon_dir / dataset_name / codec / str(bitrate) / (_Path(filename).stem + '_reconstructed.wav')

                    if not audio_path.exists() or not recon_path.exists(): continue
                    try:
                        audio, native_sr = load_audio(str(audio_path))
                        audio = peak_normalise(resample(peak_normalise(audio), native_sr, codec_sr))
                        recon, _ = load_audio(str(recon_path))
                        min_len  = min(len(audio), len(recon))
                        
                        mag_orig  = np.abs(librosa.stft(audio[:min_len], n_fft=n_fft, hop_length=hop))
                        mag_resid = np.abs(librosa.stft(recon[:min_len] - audio[:min_len], n_fft=n_fft, hop_length=hop))

                        rel_db = 20 * np.log10(mag_resid.mean(axis=1) / (mag_orig.mean(axis=1) + 1e-8))

                        for b_idx, (low_hz, high_hz, _) in enumerate(bands):
                            mask = (freqs >= low_hz) & (freqs < high_hz)
                            if mask.sum() > 0: band_residuals[b_idx].append(float(rel_db[mask].mean()))
                    except Exception:
                        continue

            means  = [np.mean(v) if v else np.nan for v in band_residuals]
            stds   = [np.std(v)  if len(v) > 1 else 0 for v in band_residuals]
            offset = (codec_idx - 0.5) * bar_width

            ax.bar(x + offset, means, width=bar_width, color=colour, alpha=0.85, label=f'{codec.upper()} {bitrate}', edgecolor='white', linewidth=0.5)
            ax.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', capsize=3, linewidth=1)

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([b[2] for b in bands], fontsize=10)
        ax.set_title(dataset_plot.replace('_', ' ').title())
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel('Mean Relative Residual (dB)')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()