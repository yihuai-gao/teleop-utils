# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from teleop_utils.utils.pose_utils import get_absolute_pose, get_relative_pose
from .iphone_client import iPhoneClient
from .iphone_command import iPhoneEvents
import numpy as np
import numpy.typing as npt
from transforms3d import quaternions



class iPhoneController:
    def __init__(
        self,
        server_address: str,
        init_pose_xyz_wxyz: npt.NDArray[np.float64],
        init_gripper_pos: float,
        arm_movement_scale: float,
        default_gripper_speed_m_per_s: float,
        gripper_max_pos: float,
        iphone_camera_in_target_eef_wxyz: npt.NDArray[np.float64],
    ):
        self.iphone_client: iPhoneClient = iPhoneClient(server_address)
        self.init_pose_xyz_wxyz: npt.NDArray[np.float64] = np.array(init_pose_xyz_wxyz)
        self.init_gripper_pos: float = init_gripper_pos
        self.arm_movement_scale: float = arm_movement_scale
        self.default_gripper_speed_m_per_s: float = default_gripper_speed_m_per_s
        self.gripper_max_pos: float = gripper_max_pos

        self.iphone_camera_in_target_eef_wxyz: npt.NDArray[np.float64] = np.array(
            iphone_camera_in_target_eef_wxyz
        )
        assert self.iphone_camera_in_target_eef_wxyz.shape == (
            4,
        ), "iphone_camera_in_target_eef_wxyz must be a (4, ) vector"

        # iphone_in_iphone_world_xyz_wxyz will be transformed to the target world frame
        self.iphone_in_iphone_world_xyz_wxyz: npt.NDArray[np.float64] = (
            init_pose_xyz_wxyz
        )
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
        self.iphone_in_iphone_world_xyz_wxyz = np.concatenate(
            (teleop_data.position_xyz, teleop_data.orientation_wxyz)
        )
        self.last_timestamp = teleop_data.xr_timestamp_ms
        self.movement_start_iphone_pose_xyz_wxyz = self.iphone_in_iphone_world_xyz_wxyz
        self.movement_start_cmd_xyz_wxyz = self.pose_cmd_xyz_wxyz

    def get_events(self):
        self.events = self.iphone_client.get_events()
        self.save_episode = False
        self.discard_episode = False
        for event in self.events:
            print(event)
            if event == iPhoneEvents.SAVE_EPISODE:
                self.save_episode = True
                self.session_started = True
                self.reset_cmd()
            elif event == iPhoneEvents.DISCARD_EPISODE:
                self.discard_episode = True
                self.session_started = True
                self.reset_cmd()
            elif event == iPhoneEvents.TOGGLE_MOVEMENT:
                self.in_movement = not self.in_movement
                self.session_started = True
                if self.in_movement:
                    self.reset_movement()
            elif event == iPhoneEvents.START_SESSION:
                self.session_started = True
            elif event == iPhoneEvents.END_SESSION:
                self.reset_cmd()
                self.reset_movement()
                self.session_started = False
        return self.session_started, self.save_episode, self.discard_episode

    def get_cmd(self):
        teleop_data = self.iphone_client.get_latest_pose()
        if teleop_data is not None:

            self.iphone_in_iphone_world_xyz_wxyz = np.concatenate(
                (teleop_data.position_xyz, teleop_data.orientation_wxyz)
            )
            # print(teleop_data.position_xyz, teleop_data.orientation_wxyz)
            if self.in_movement:
                assert self.movement_start_iphone_pose_xyz_wxyz is not None
                movement_pose_in_iphone_camera_xyz_wxyz = get_relative_pose(
                    self.iphone_in_iphone_world_xyz_wxyz,
                    self.movement_start_iphone_pose_xyz_wxyz,
                )
                # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
                x, y, z, qw, qx, qy, qz = movement_pose_in_iphone_camera_xyz_wxyz

                # Robot arm view
                # x, y, z = z, x, y
                # qw, qx, qy, qz = qw, qx, qz, -qy

                # Front view
                x, y, z = -z, -x, y
                qw, qx, qy, qz = qw, -qx, -qz, -qy
                movement_pose_in_iphone_camera_xyz_wxyz = np.array(
                    [x, y, z, qw, qx, qy, qz]
                )
                movement_pose_in_iphone_camera_xyz_wxyz[:3] *= self.arm_movement_scale
                self.pose_cmd_xyz_wxyz = get_absolute_pose(
                    self.movement_start_cmd_xyz_wxyz,
                    movement_pose_in_iphone_camera_xyz_wxyz,
                )
                self.gripper_pos_cmd += (
                    self.default_gripper_speed_m_per_s
                    / 1e3
                    * (teleop_data.xr_timestamp_ms - self.last_timestamp)
                    * teleop_data.gripper_speed
                )
                print(teleop_data.xr_timestamp_ms - self.last_timestamp)
                self.gripper_pos_cmd = np.clip(
                    self.gripper_pos_cmd, 0.0, self.gripper_max_pos
                )
            self.last_timestamp = teleop_data.xr_timestamp_ms

        return self.in_movement, self.pose_cmd_xyz_wxyz, self.gripper_pos_cmd
