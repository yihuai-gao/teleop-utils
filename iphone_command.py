from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any
import numpy as np
import numpy.typing as npt


class iPhoneCommandType(IntEnum):
    START_EPISODE = auto()
    END_EPISODE = auto()
    START_MOVEMENT = auto()
    END_MOVEMENT = auto()
    SWITCH_ROBOT = auto()
    SEND_DATA = auto()


@dataclass
class iPhoneCommand:
    command_type: iPhoneCommandType
    timestamp: float
    position_xyz: npt.NDArray[np.float64]
    orientation_wxyz: npt.NDArray[np.float64]
    gripper: float
