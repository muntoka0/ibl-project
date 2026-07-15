"""
Pure Python GLM-HMM-compatible subset of ssm.

This package intentionally implements only the input-driven-observation HMM
path needed for the IBL GLM-HMM notebooks.  It avoids the old Cython/C++
backend that breaks under some Python 3.11 installs.
"""

from .hmm import HMM
from .init_state_distns import InitialStateDistribution
from .observations import InputDrivenObservations
from .transitions import StationaryTransitions

__version__ = "0.1.0-glmhmm-python311"

__all__ = [
    "HMM",
    "InitialStateDistribution",
    "InputDrivenObservations",
    "StationaryTransitions",
]
