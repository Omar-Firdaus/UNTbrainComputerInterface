
"""
csa.py — Compressed Spectral Array (CSA) computation.

Public API
----------
compute_csa(signal, fs, ...) -> CSAResult
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend, welch


@dataclass
class CSAResult:
    """Output of :func:`compute_csa`.

    Attributes
    ----------
    power_matrix : ndarray, shape (n_channels, n_epochs, n_freqs)
        Log-power in dB (10 * log10(PSD + eps)).
    freqs : ndarray, shape (n_freqs,)
        Frequency axis in Hz.
    times : ndarray, shape (n_epochs,)
        Epoch centre times in seconds.
    epoch_indices : ndarray, shape (n_epochs, 2)
        Sample [start, stop) for each epoch.
    fs : float
        Sampling frequency in Hz.
    epoch_sec : float
        Epoch length in seconds as requested.
    """

    power_matrix: np.ndarray
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
    """Compute a Compressed Spectral Array from a multi-channel EEG signal.

    Parameters
    ----------
    signal : ndarray, shape (n_channels, n_samples)
        EEG signal in µV (or any consistent unit).
    fs : float
        Sampling frequency in Hz.  Must be > 0.
    epoch_sec : float
        Epoch length in seconds.
    overlap : float
        Fractional overlap between successive epochs in [0, 1).
    fmax : float
        Maximum frequency to include in the output (Hz).  Must be ≤ fs/2.
    method : str
        PSD method — only ``"welch"`` is currently supported.

    Returns
    
    -------
    CSAResult
        Contains the log-power matrix, frequency axis, time axis, and
        epoch sample indices.

    Raises
    ------
    ValueError
        If ``fs <= 0``, the epoch is longer than the signal, or
        ``fmax > fs / 2``.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")
    if epoch_sec <= 0:
        raise ValueError(f"epoch_sec must be > 0, got {epoch_sec}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if fmax > fs / 2:
        raise ValueError(
            f"fmax ({fmax} Hz) exceeds Nyquist ({fs / 2} Hz). "
            "Lower fmax or increase fs."
        )
    if method != "welch":
        raise ValueError(f"Unsupported method '{method}'. Only 'welch' is supported.")

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError(
            f"signal must be 2-D (n_channels, n_samples), got shape {signal.shape}"
        )

    n_channels, n_samples = signal.shape
    epoch_len = int(epoch_sec * fs)

    if epoch_len > n_samples:
        raise ValueError(
            f"Epoch length ({epoch_len} samples = {epoch_sec} s) exceeds "
            f"signal length ({n_samples} samples = {n_samples / fs:.2f} s)."
        )

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------
    nan_fraction = np.isnan(signal).mean()
    if nan_fraction > 0:
        if nan_fraction > 0.10:
            warnings.warn(
                f"{nan_fraction:.1%} of samples are NaN — zeroing them. "
                "Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"{nan_fraction:.1%} of samples are NaN — zeroing them.",
                RuntimeWarning,
                stacklevel=2,
            )
        signal = np.where(np.isnan(signal), 0.0, signal)

    # ------------------------------------------------------------------
    # Build epoch grid
    # ------------------------------------------------------------------
    step = int(epoch_len * (1.0 - overlap))
    if step < 1:
        step = 1

    starts = np.arange(0, n_samples - epoch_len + 1, step)
    n_epochs = len(starts)
    stops = starts + epoch_len

    # epochs: shape (n_channels, n_epochs, epoch_len) — copy to allow detrend
    epochs = np.stack(
        [signal[:, s:s + epoch_len] for s in starts], axis=1
    )  # (C, E, L)

    # ------------------------------------------------------------------
    # Detrend + window
    # ------------------------------------------------------------------
    epochs = detrend(epochs, axis=-1)  # linear detrend per epoch per channel
    window = np.hanning(epoch_len)     # Hann window
    epochs *= window[np.newaxis, np.newaxis, :]

    # ------------------------------------------------------------------
    # Welch PSD — computed per channel, vectorised over epochs
    # ------------------------------------------------------------------
    all_freqs: np.ndarray | None = None
    power_list: list[np.ndarray] = []

    for ch in range(n_channels):
        # epochs[ch]: shape (n_epochs, epoch_len)
        ch_freqs, ch_psd = welch(
            epochs[ch],
            fs=fs,
            window="hann",
            nperseg=epoch_len,
            noverlap=0,      # already manually overlapped
            axis=-1,
        )  # ch_psd: (n_epochs, n_freqs_all)

        if all_freqs is None:
            all_freqs = ch_freqs

        power_list.append(ch_psd)

    psd_all = np.stack(power_list, axis=0)  # (C, E, F_all)

    # ------------------------------------------------------------------
    # Trim to fmax and convert to log-power (dB)
    # ------------------------------------------------------------------
    assert all_freqs is not None
    freq_mask = all_freqs <= fmax
    freqs = all_freqs[freq_mask]
    psd_trimmed = psd_all[:, :, freq_mask]

    eps = 1e-12
    power_matrix = 10.0 * np.log10(psd_trimmed + eps)  # (C, E, F)

    # ------------------------------------------------------------------
    # Build time and index axes
    # ------------------------------------------------------------------
    times = (starts + epoch_len / 2.0) / fs   # epoch centre in seconds
    epoch_indices = np.column_stack([starts, stops])  # (E, 2)

    return CSAResult(
        power_matrix=power_matrix,
        freqs=freqs,
        times=times,
        epoch_indices=epoch_indices,
        fs=float(fs),
        epoch_sec=float(epoch_sec),
    )
