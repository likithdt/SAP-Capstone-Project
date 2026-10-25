"""
Time-Domain Feature Extraction for Industrial Machinery Vibration and Telemetry.
Extracts statistical metrics capturing vibration energy, shock severity, distribution asymmetry, and impulsiveness.
"""

from typing import Dict, Union, Optional
import numpy as np


def compute_time_domain_features(
    signal: np.ndarray,
    prefix: str = "vibration"
) -> Dict[str, float]:
    """
    Extract comprehensive time-domain statistical features from a 1D signal window.
    
    Features computed:
      - RMS: Overall vibration energy
      - Peak / Peak-to-Peak: Maximum absolute amplitude and full excursion
      - Kurtosis: Impulsiveness / sharpness (Pearson, normal distribution = 3.0)
      - Crest Factor: Peak severity relative to RMS
      - Skewness: Asymmetry of the vibration distribution
      - Standard Deviation: Dynamic variance / operational stability
      - Mean: DC offset
      - Impulse Factor: Peak / mean absolute amplitude
      - Margin Factor: Peak / square of mean square-root amplitude
      - Shape Factor: RMS / mean absolute amplitude
      - Zero Crossing Rate: Frequency of sign changes per sample
    """
    clean = signal[~np.isnan(signal)] if np.isnan(signal).any() else signal
    n = len(clean)

    if n == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_peak": 0.0,
            f"{prefix}_peak_to_peak": 0.0,
            f"{prefix}_crest_factor": 0.0,
            f"{prefix}_kurtosis": 0.0,
            f"{prefix}_skewness": 0.0,
            f"{prefix}_impulse_factor": 0.0,
            f"{prefix}_margin_factor": 0.0,
            f"{prefix}_shape_factor": 0.0,
            f"{prefix}_zero_crossing_rate": 0.0,
        }

    mean_val = float(np.mean(clean))
    std_val = float(np.std(clean))
    rms_val = float(np.sqrt(np.mean(clean**2)))
    peak_val = float(np.max(np.abs(clean)))
    peak_to_peak_val = float(np.ptp(clean))

    # Crest factor
    crest_factor = (peak_val / rms_val) if rms_val > 1e-9 else 0.0

    # Kurtosis & Skewness
    if std_val > 1e-9:
        centered = clean - mean_val
        kurtosis_val = float(np.mean(centered**4) / (std_val**4))
        skewness_val = float(np.mean(centered**3) / (std_val**3))
    else:
        kurtosis_val = 0.0
        skewness_val = 0.0

    # Impulse, Margin, and Shape factors
    abs_clean = np.abs(clean)
    mean_abs = float(np.mean(abs_clean))

    impulse_factor = (peak_val / mean_abs) if mean_abs > 1e-9 else 0.0
    mean_sqrt = float(np.mean(np.sqrt(abs_clean)))
    margin_factor = (peak_val / (mean_sqrt**2)) if mean_sqrt > 1e-9 else 0.0
    shape_factor = (rms_val / mean_abs) if mean_abs > 1e-9 else 0.0

    # Zero crossing rate
    signs = np.sign(clean)
    signs[signs == 0] = 1
    zero_crossings = np.sum(np.diff(signs) != 0)
    zcr = float(zero_crossings / (n - 1)) if n > 1 else 0.0

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_peak": peak_val,
        f"{prefix}_peak_to_peak": peak_to_peak_val,
        f"{prefix}_crest_factor": crest_factor,
        f"{prefix}_kurtosis": kurtosis_val,
        f"{prefix}_skewness": skewness_val,
        f"{prefix}_impulse_factor": impulse_factor,
        f"{prefix}_margin_factor": margin_factor,
        f"{prefix}_shape_factor": shape_factor,
        f"{prefix}_zero_crossing_rate": zcr,
    }


def compute_temperature_features(temp_signal: np.ndarray) -> Dict[str, float]:
    """Extract summary and trend features for thermal telemetry."""
    clean = temp_signal[~np.isnan(temp_signal)] if np.isnan(temp_signal).any() else temp_signal
    if len(clean) == 0:
        return {"temp_mean": 0.0, "temp_std": 0.0, "temp_min": 0.0, "temp_max": 0.0, "temp_slope": 0.0}

    # Trend slope via linear regression over index
    x = np.arange(len(clean))
    if len(clean) > 1 and np.std(x) > 1e-9:
        slope = float(np.polyfit(x, clean, deg=1)[0])
    else:
        slope = 0.0

    return {
        "temp_mean": float(np.mean(clean)),
        "temp_std": float(np.std(clean)),
        "temp_min": float(np.min(clean)),
        "temp_max": float(np.max(clean)),
        "temp_slope": slope,
    }


def compute_current_features(current_signal: np.ndarray, fs: float = 1000.0) -> Dict[str, float]:
    """Extract electrical current metrics including DC mean, RMS, and 100 Hz ripple."""
    clean = current_signal[~np.isnan(current_signal)] if np.isnan(current_signal).any() else current_signal
    if len(clean) == 0:
        return {
            "current_mean": 0.0,
            "current_rms": 0.0,
            "current_std": 0.0,
            "current_peak_to_peak": 0.0,
            "current_100hz_ripple": 0.0,
        }

    mean_val = float(np.mean(clean))
    rms_val = float(np.sqrt(np.mean(clean**2)))
    std_val = float(np.std(clean))
    ptp_val = float(np.ptp(clean))

    # Compute 100 Hz ripple amplitude via FFT
    ac_component = clean - mean_val
    n = len(ac_component)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_amps = np.abs(np.fft.rfft(ac_component)) * (2.0 / n)
    idx_100 = np.argmin(np.abs(freqs - 100.0))
    ripple_100 = float(fft_amps[idx_100]) if len(fft_amps) > idx_100 else 0.0

    return {
        "current_mean": mean_val,
        "current_rms": rms_val,
        "current_std": std_val,
        "current_peak_to_peak": ptp_val,
        "current_100hz_ripple": ripple_100,
    }
