"""Offline target adapter for the original-size submission candidate.

The user allocation, reserve, pair strategy and execution risk caps stay the
same. This module is not imported by either running competition service.
"""
import numpy as np

MULTIPLIER=1.44117372816997


def transform(default_weights):
    weights=np.asarray(default_weights,float)
    if weights.shape[-1:]!=(10,) or not np.isfinite(weights).all():
        raise ValueError('Expected finite targets for ten assets')
    return weights*MULTIPLIER
