"""
Frequency-Domain (FFT) and Envelope Spectrum Feature Extraction.
Transforms raw vibration time-series into diagnostic frequency features:
  - Dominant frequency and top spectral peaks
  - Spectral centroid, spread, skewness, and kurtosis
  - Spectral entropy and spectral flatness
  - Diagnostic band energies (Sub-sync, 1X, 2X, 3X, BPFO, High-freq)
  - Total Harmonic Distortion (THD)
  - Hilbert transform Envelope-spectrum BPFO defect peak amplitude
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.signal import hilbert


def compute_rfft_spectrum(
    signal: np.ndarray,
    fs: float = 1000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Real-valued FFT frequency vector, amplitude spectrum, and power spectrum.
    
    Returns:
      freqs: Frequency vector in Hz (from 0 to Nyquist fs/2)
      amplitudes: Linear amplitude spectrum (scaled to physical units)
      power_spectrum: Power spectrum density P(f)
    """
    clean = signal[~np.isnan(signal)] if np.isnan(signal).any() else signal
    n = len(clean)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_complex = np.fft.rfft(clean)
    amplitudes = np.abs(fft_complex) * (2.0 / n)
    amplitudes[0] /= 2.0  # Adjust DC component
    power_spectrum = (np.abs(fft_complex) ** 2) / (n * fs)
    return freqs, amplitudes, power_spectrum


def compute_spectral_moments(
    freqs: np.ndarray,
    power: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistical spectral moments (Centroid, Spread, Skewness, Kurtosis).
    Excludes DC component (0 Hz) for mechanical relevance.
    """
    if len(freqs) <= 1 or np.sum(power[1:]) <= 1e-12:
        return {
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_skewness": 0.0,
            "spectral_kurtosis": 0.0,
        }

    f = freqs[1:]
    p = power[1:]
    total_power = np.sum(p)
    weights = p / total_power

    # 1. Spectral Centroid
    centroid = float(np.sum(f * weights))

    # 2. Spectral Spread (Standard Deviation of Spectrum)
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * weights)))

    # 3. Spectral Skewness
    if spread > 1e-9:
        skewness = float(np.sum(((f - centroid) ** 3) * weights) / (spread ** 3))
        kurtosis = float(np.sum(((f - centroid) ** 4) * weights) / (spread ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    return {
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_skewness": skewness,
        "spectral_kurtosis": kurtosis,
    }


def compute_spectral_entropy_and_flatness(
    power: np.ndarray
) -> Dict[str, float]:
    """
    Compute Spectral Entropy (Shannon entropy of normalized power) and
    Spectral Flatness (Wiener entropy: Geometric mean / Arithmetic mean).
    """
    if len(power) <= 1:
        return {"spectral_entropy": 0.0, "spectral_flatness": 0.0}

    p = power[1:]  # Exclude DC
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return {"spectral_entropy": 0.0, "spectral_flatness": 0.0}

    # Normalized distribution
    prob = p / p_sum
    prob_nonzero = prob[prob > 0]
    shannon_entropy = -float(np.sum(prob_nonzero * np.log2(prob_nonzero)))
    # Max possible entropy for uniform spectrum = log2(N)
    max_entropy = np.log2(len(p))
    norm_entropy = (shannon_entropy / max_entropy) if max_entropy > 0 else 0.0

    # Spectral flatness (Wiener entropy): 0 for pure tone, 1 for white noise
    geom_mean = float(np.exp(np.mean(np.log(p + 1e-15))))
    arith_mean = float(np.mean(p))
    flatness = (geom_mean / arith_mean) if arith_mean > 1e-12 else 0.0

    return {
        "spectral_entropy": norm_entropy,
        "spectral_flatness": min(max(flatness, 0.0), 1.0),
    }


def compute_band_energies(
    freqs: np.ndarray,
    power: np.ndarray,
    f0: float = 30.0,
    bpfo: float = 107.4
) -> Dict[str, float]:
    """
    Compute integrated energy across critical industrial diagnostic bands:
      - Sub-synchronous band: [0, 0.85 * f0] (e.g. 0-25 Hz)
      - 1X fundamental band: [0.85 * f0, 1.15 * f0] (e.g. 25-35 Hz) -> Rotor Imbalance
      - 2X harmonic band: [1.85 * f0, 2.15 * f0] (e.g. 55-65 Hz) -> Misalignment
      - 3X harmonic band: [2.85 * f0, 3.15 * f0] (e.g. 85-95 Hz) -> Misalignment
      - BPFO defect band: [bpfo - 5, bpfo + 5] (e.g. 102.4-112.4 Hz) -> Bearing Wear
      - High-frequency band: [120, fs/2] -> Shock impacts, friction
    """
    if len(freqs) == 0:
        return {
            "band_sub_sync_energy": 0.0,
            "band_1x_energy": 0.0,
            "band_2x_energy": 0.0,
            "band_3x_energy": 0.0,
            "band_bpfo_energy": 0.0,
            "band_high_freq_energy": 0.0,
            "harmonic_energy_ratio": 0.0,
            "total_harmonic_distortion": 0.0,
        }

    def band_power(f_min: float, f_max: float) -> float:
        mask = (freqs >= f_min) & (freqs <= f_max)
        return float(np.sum(power[mask])) if np.any(mask) else 0.0

    p_sub = band_power(0.5, 0.85 * f0)
    p_1x = band_power(0.85 * f0, 1.15 * f0)
    p_2x = band_power(1.85 * f0, 2.15 * f0)
    p_3x = band_power(2.85 * f0, 3.15 * f0)
    p_bpfo = band_power(bpfo - 5.0, bpfo + 5.0)
    p_high = band_power(120.0, freqs[-1])

    # Harmonic ratios for misalignment vs imbalance differentiation
    harmonic_ratio = ((p_2x + p_3x) / p_1x) if p_1x > 1e-9 else 0.0
    thd = (np.sqrt(p_2x + p_3x) / np.sqrt(p_1x)) if p_1x > 1e-9 else 0.0

    return {
        "band_sub_sync_energy": p_sub,
        "band_1x_energy": p_1x,
        "band_2x_energy": p_2x,
        "band_3x_energy": p_3x,
        "band_bpfo_energy": p_bpfo,
        "band_high_freq_energy": p_high,
        "harmonic_energy_ratio": harmonic_ratio,
        "total_harmonic_distortion": thd,
    }


def compute_envelope_bpfo_features(
    signal: np.ndarray,
    fs: float = 1000.0,
    bpfo: float = 107.4
) -> Dict[str, float]:
    """
    Extract Hilbert transform envelope spectrum features for bearing defect analysis.
    Demodulates high-frequency resonance impulses down to defect impact repetition rate (BPFO).
    """
    clean = signal[~np.isnan(signal)] if np.isnan(signal).any() else signal
    n = len(clean)
    if n < 32:
        return {"envelope_bpfo_peak_amp": 0.0, "envelope_bpfo_ratio": 0.0}

    # Analytic signal via Hilbert transform
    analytic = hilbert(clean)
    envelope = np.abs(analytic)
    # Remove envelope DC mean
    envelope_ac = envelope - np.mean(envelope)

    # Envelope FFT
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    env_fft = np.abs(np.fft.rfft(envelope_ac)) * (2.0 / n)

    # BPFO peak window [bpfo - 3 Hz, bpfo + 3 Hz]
    bpfo_mask = (freqs >= (bpfo - 3.0)) & (freqs <= (bpfo + 3.0))
    bpfo_peak = float(np.max(env_fft[bpfo_mask])) if np.any(bpfo_mask) else 0.0

    # Background noise floor in envelope spectrum (excluding low freqs < 10 Hz)
    noise_mask = freqs > 10.0
    median_noise = float(np.median(env_fft[noise_mask])) if np.any(noise_mask) else 1e-9
    envelope_ratio = (bpfo_peak / median_noise) if median_noise > 1e-9 else 0.0

    return {
        "envelope_bpfo_peak_amp": bpfo_peak,
        "envelope_bpfo_ratio": envelope_ratio,
    }


def compute_frequency_domain_features(
    signal: np.ndarray,
    fs: float = 1000.0,
    f0: float = 30.0,
    bpfo: float = 107.4,
    prefix: str = "vibration"
) -> Dict[str, float]:
    """
    Comprehensive frequency-domain feature extraction pipeline for a 1D vibration signal.
    """
    freqs, amplitudes, power = compute_rfft_spectrum(signal, fs=fs)
    if len(freqs) <= 1:
        return {}

    # Dominant frequency (excluding DC bin)
    dom_idx = int(np.argmax(amplitudes[1:])) + 1
    dom_freq = float(freqs[dom_idx])
    dom_amp = float(amplitudes[dom_idx])

    # Spectral moments
    moments = compute_spectral_moments(freqs, power)
    # Spectral entropy and flatness
    entropy_flatness = compute_spectral_entropy_and_flatness(power)
    # Diagnostic band energies
    band_energies = compute_band_energies(freqs, power, f0=f0, bpfo=bpfo)
    # Envelope BPFO features
    envelope_feat = compute_envelope_bpfo_features(signal, fs=fs, bpfo=bpfo)

    result = {
        f"{prefix}_dominant_freq_hz": dom_freq,
        f"{prefix}_dominant_amp": dom_amp,
    }

    for k, v in moments.items():
        result[f"{prefix}_{k}"] = v
    for k, v in entropy_flatness.items():
        result[f"{prefix}_{k}"] = v
    for k, v in band_energies.items():
        result[f"{prefix}_{k}"] = v
    for k, v in envelope_feat.items():
        result[f"{prefix}_{k}"] = v

    return result
