"""
miburi is the implementation of MIBURI: Towards Expressive Interactive Gesture
Synthesis (CVPR 2026).

Built on top of Kyutai's Moshi/Mimi (https://github.com/kyutai-labs/moshi);
portions of the code are adapted from Moshi and from Audiocraft (Meta Platforms),
see LICENSE for details.
"""

# flake8: noqa
from . import conditioners
from . import models
from . import modules
from . import quantization
from . import utils

__version__ = "0.1.0a1"
