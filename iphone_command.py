from dataclasses import dataclass
from enum import IntEnum, auto
import numpy as np
import numpy.typing as npt


class iPhoneEvents(IntEnum):
    START_SESSION = auto()
    PREPARE_END_SESSION = auto()
    END_SESSION = auto()
    START_EPISODE = auto()
    END_EPISODE = auto()
    START_MOVEMENT = auto()
    END_MOVEMENT = auto()
    RESET_ENV = auto()
    SWITCH_ROBOT = auto()


@dataclass
class TeleopData:
    timestamp: float
    position_xyz: npt.NDArray[np.float64]
    orientation_wxyz: npt.NDArray[np.float64]
    gripper: float
