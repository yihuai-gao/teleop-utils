# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from dataclasses import dataclass
from enum import IntEnum, auto
import numpy as np
import numpy.typing as npt


class iPhoneEvents(IntEnum):
    START_SESSION = auto()
    PREPARE_END_SESSION = auto()
    END_SESSION = auto()
    TOGGLE_MOVEMENT = auto()
    SAVE_EPISODE = auto()
    DISCARD_EPISODE = auto()


@dataclass
class TeleopData:
    timestamp: float
    xr_timestamp_ms: float
    position_xyz: npt.NDArray[np.float64]
    orientation_wxyz: npt.NDArray[np.float64]
    gripper_speed: float
