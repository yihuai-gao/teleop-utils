import numpy as np
import numpy.typing as npt
from transforms3d import quaternions


def get_absolute_pose(
    init_pose_xyz_wxyz: npt.NDArray[np.float64],
    relative_pose_xyz_wxyz: npt.NDArray[np.float64],
):
    """The new pose is in the same frame of reference as the initial pose"""
    new_pose_xyz_wxyz = np.zeros(7, dtype=np.float64)
    relative_pos_in_init_frame_as_quat_wxyz = np.zeros(4, dtype=np.float64)
    relative_pos_in_init_frame_as_quat_wxyz[1:] = relative_pose_xyz_wxyz[:3]
    init_rot_qinv = quaternions.qconjugate(init_pose_xyz_wxyz[3:])
    relative_pos_in_world_frame_as_quat_wxyz = quaternions.qmult(
        quaternions.qmult(
            init_pose_xyz_wxyz[3:], relative_pos_in_init_frame_as_quat_wxyz
        ),
        init_rot_qinv,
    )
    new_pose_xyz_wxyz[:3] = (
        init_pose_xyz_wxyz[:3] + relative_pos_in_world_frame_as_quat_wxyz[1:]
    )
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
    relative_pos_in_world_frame_as_quat_wxyz = np.zeros(4, dtype=np.float64)
    relative_pos_in_world_frame_as_quat_wxyz[1:] = (
        new_pose_xyz_wxyz[:3] - init_pose_xyz_wxyz[:3]
    )
    init_rot_qinv = quaternions.qconjugate(init_pose_xyz_wxyz[3:])
    relative_pose_xyz_wxyz[:3] = quaternions.qmult(
        quaternions.qmult(init_rot_qinv, relative_pos_in_world_frame_as_quat_wxyz),
        init_pose_xyz_wxyz[3:],
    )[1:]
    relative_pose_xyz_wxyz[3:] = quaternions.qmult(init_rot_qinv, new_pose_xyz_wxyz[3:])
    return relative_pose_xyz_wxyz


def invert_pose(pose_xyz_wxyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    qinv = quaternions.qconjugate(pose_xyz_wxyz[3:])
    pos_quat_wxyz = np.zeros(4, dtype=np.float64)
    pos_quat_wxyz[1:] = pose_xyz_wxyz[:3]
    rotated_pos = quaternions.qmult(
        quaternions.qmult(qinv, pos_quat_wxyz),
        pose_xyz_wxyz[3:],
    )
    inverted_pose = np.zeros(7, dtype=np.float64)
    inverted_pose[:3] = -rotated_pos[1:]
    inverted_pose[3:] = qinv
    return inverted_pose
