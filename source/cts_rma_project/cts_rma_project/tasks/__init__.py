# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

from .shared   import *   # noqa  — GO2 shared rewards/obs
from .baseline import *   # noqa  — GO2 baseline
from .rma      import *   # noqa  — GO2 RMA
from .cts      import *   # noqa  — GO2 CTS
from .one_leg  import *   # noqa  — one-leg hopper

_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
