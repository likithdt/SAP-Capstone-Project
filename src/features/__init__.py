"""
Feature Engineering module for industrial telemetry and prognostics.
"""

from src.features.time_domain import (
    compute_time_domain_features,
    compute_temperature_features,
    compute_current_features,
)
from src.features.frequency_domain import (
    compute_frequency_domain_features,
    compute_rfft_spectrum,
    compute_spectral_moments,
    compute_band_energies,
    compute_envelope_bpfo_features,
)
from src.features.pipeline import FeatureExtractionPipeline

__all__ = [
    "FeatureExtractionPipeline",
    "compute_time_domain_features",
    "compute_temperature_features",
    "compute_current_features",
    "compute_frequency_domain_features",
    "compute_rfft_spectrum",
    "compute_spectral_moments",
    "compute_band_energies",
    "compute_envelope_bpfo_features",
]
