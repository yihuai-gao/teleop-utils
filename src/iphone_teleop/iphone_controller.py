from .iphone_client import iPhoneClient
from .iphone_command import iPhoneEvents
import numpy as np
import numpy.typing as npt
from transforms3d import quaternions


def get_new_pose(
    init_pose_xyz_wxyz: npt.NDArray[np.float64],
    relative_pose_xyz_wxyz: npt.NDArray[np.float64],
):
    """The new pose is in the same frame of reference as the initial pose"""
    new_pose_xyz_wxyz = np.zeros(7, dtype=np.float64)
    new_pose_xyz_wxyz[:3] = init_pose_xyz_wxyz[:3] + relative_pose_xyz_wxyz[:3]
    new_pose_xyz_wxyz[3:] = quaternions.qmult(
        init_pose_xyz_wxyz[3:], relative_pose_xyz_wxyz[3:]
    )
    return new_pose_xyz_wxyz


def get_relative_pose(
    new_pose_xyz_wxyz: npt.NDArray[np.float64],
    init_pose_xyz_wxyz: npt.NDArray[np.float64],
):
    """The two poses are in the same frame of reference"""
    relative_pose_xyz_wxyz = np.zeros(7, dtype=np.float64)
    relative_pose_xyz_wxyz[:3] = new_pose_xyz_wxyz[:3] - init_pose_xyz_wxyz[:3]
    relative_pose_xyz_wxyz[3:] = quaternions.qmult(
        quaternions.qinverse(init_pose_xyz_wxyz[3:]), new_pose_xyz_wxyz[3:]
    )
    return relative_pose_xyz_wxyz


class iPhoneController:
    def __init__(
        self,
        server_address: str,
        init_pose_xyz_wxyz: npt.NDArray[np.float64],
        init_gripper_pos: float,
        arm_movement_scale: float,
        default_gripper_speed: float,
        gripper_max_pos: float,
    ):
        self.iphone_client: iPhoneClient = iPhoneClient(server_address)
        self.init_pose_xyz_wxyz: npt.NDArray[np.float64] = init_pose_xyz_wxyz
        self.init_gripper_pos: float = init_gripper_pos
        self.arm_movement_scale: float = arm_movement_scale
        self.default_gripper_speed: float = default_gripper_speed
        self.gripper_max_pos: float = gripper_max_pos

        self.latest_iphone_pose_xyz_wxyz: npt.NDArray[np.float64] = init_pose_xyz_wxyz
        self.pose_cmd_xyz_wxyz: npt.NDArray[np.float64] = init_pose_xyz_wxyz
        self.gripper_pos_cmd: float = init_gripper_pos
        self.in_movement: bool = False
        self.events: list[iPhoneEvents] = []
        self.save_episode: bool = False
        self.discard_episode: bool = False
        self.session_started: bool = False

        self.movement_start_iphone_pose_xyz_wxyz: npt.NDArray[np.float64] | None = None
        self.movement_start_cmd_xyz_wxyz: npt.NDArray[np.float64] = (
            self.pose_cmd_xyz_wxyz
        )
        self.last_timestamp: float = 0.0

    def reset_cmd(self):
        self.pose_cmd_xyz_wxyz = self.init_pose_xyz_wxyz
        self.gripper_pos_cmd = self.init_gripper_pos
        self.movement_start_iphone_pose_xyz_wxyz = None
        self.last_timestamp = 0.0
        self.in_movement = False

    def reset_movement(self):
        teleop_data = self.iphone_client.get_latest_pose()
        assert teleop_data is not None, "No pose data received when resetting movement"
        self.latest_iphone_pose_xyz_wxyz = np.concatenate(
            (teleop_data.position_xyz, teleop_data.orientation_wxyz)
        )
        self.last_timestamp = teleop_data.xr_timestamp
        self.movement_start_iphone_pose_xyz_wxyz = self.latest_iphone_pose_xyz_wxyz
        self.movement_start_cmd_xyz_wxyz = self.pose_cmd_xyz_wxyz

    def get_events(self):
        self.events = self.iphone_client.get_events()
        self.save_episode = False
        self.discard_episode = False
        for event in self.events:
            print(event)
            if event == iPhoneEvents.SAVE_EPISODE:
                self.save_episode = True
                self.reset_cmd()
            elif event == iPhoneEvents.DISCARD_EPISODE:
                self.discard_episode = True
                self.reset_cmd()
            elif event == iPhoneEvents.TOGGLE_MOVEMENT:
                self.in_movement = not self.in_movement
                if self.in_movement:
                    self.reset_movement()
            elif event == iPhoneEvents.START_SESSION:
                self.session_started = True
            elif event == iPhoneEvents.END_SESSION:
                self.session_started = False
        return self.session_started, self.save_episode, self.discard_episode

    def get_cmd(self):
        teleop_data = self.iphone_client.get_latest_pose()
        if teleop_data is not None:
            self.latest_iphone_pose_xyz_wxyz = np.concatenate(
                (teleop_data.position_xyz, teleop_data.orientation_wxyz)
            )
            if self.in_movement:
                assert self.movement_start_iphone_pose_xyz_wxyz is not None
                relative_pose_xyz_wxyz = get_relative_pose(
                    self.latest_iphone_pose_xyz_wxyz,
                    self.movement_start_iphone_pose_xyz_wxyz,
                )
                self.pose_cmd_xyz_wxyz = get_new_pose(
                    self.movement_start_cmd_xyz_wxyz, relative_pose_xyz_wxyz
                )
                self.gripper_pos_cmd += self.default_gripper_speed * (
                    teleop_data.xr_timestamp - self.last_timestamp
                )
                self.gripper_pos_cmd = np.clip(
                    self.gripper_pos_cmd, 0.0, self.gripper_max_pos
                )
            self.last_timestamp = teleop_data.xr_timestamp

        return self.in_movement, self.pose_cmd_xyz_wxyz, self.gripper_pos_cmd
