from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend, welch


@dataclass
class CSAResult:
    power_matrix: np.ndarray   # (n_channels, n_epochs, n_freqs) log-power dB
    freqs: np.ndarray
    times: np.ndarray
    epoch_indices: np.ndarray
    fs: float
    epoch_sec: float


def compute_csa(
    signal: np.ndarray,
    fs: float,
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
    fmax: float = 30.0,
    method: str = "welch",
) -> CSAResult:
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    if epoch_sec <= 0:
        raise ValueError(f"epoch_sec must be > 0, got {epoch_sec}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if fmax > fs / 2:
        raise ValueError(f"fmax ({fmax} Hz) exceeds Nyquist ({fs / 2} Hz).")
    if method != "welch":
        raise ValueError(f"Unsupported method '{method}'.")

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (n_channels, n_samples), got shape {signal.shape}")

    n_channels, n_samples = signal.shape
    epoch_len = int(epoch_sec * fs)

    if epoch_len > n_samples:
        raise ValueError(
            f"Epoch length ({epoch_len} samples) exceeds signal length ({n_samples} samples)."
        )

    nan_fraction = np.isnan(signal).mean()
    if nan_fraction > 0:
        warnings.warn(f"{nan_fraction:.1%} of samples are NaN — zeroing them.", RuntimeWarning, stacklevel=2)
        signal = np.where(np.isnan(signal), 0.0, signal)

    step = max(1, int(epoch_len * (1.0 - overlap)))
    starts = np.arange(0, n_samples - epoch_len + 1, step)
    stops = starts + epoch_len

    epochs = np.stack([signal[:, s:s + epoch_len] for s in starts], axis=1)
    epochs = detrend(epochs, axis=-1)
    window = np.hanning(epoch_len)
    epochs *= window[np.newaxis, np.newaxis, :]

    all_freqs = None
    power_list = []

    for ch in range(n_channels):
        ch_freqs, ch_psd = welch(epochs[ch], fs=fs, window="hann", nperseg=epoch_len, noverlap=0, axis=-1)
        if all_freqs is None:
            all_freqs = ch_freqs
        power_list.append(ch_psd)

    psd_all = np.stack(power_list, axis=0)

    freq_mask = all_freqs <= fmax
    freqs = all_freqs[freq_mask]
    psd_trimmed = psd_all[:, :, freq_mask]

    power_matrix = 10.0 * np.log10(psd_trimmed + 1e-12)
    times = (starts + epoch_len / 2.0) / fs
    epoch_indices = np.column_stack([starts, stops])

    return CSAResult(
        power_matrix=power_matrix,
        freqs=freqs,
        times=times,
        epoch_indices=epoch_indices,
        fs=float(fs),
        epoch_sec=float(epoch_sec),
    )
