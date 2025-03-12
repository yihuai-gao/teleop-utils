# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import scipy.interpolate as interp
import click
import numpy.typing as npt
import pickle
import av
from datetime import datetime, timedelta
import os


def interpolate_mocap_poses(
    poses: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.float64],
    new_timestamps: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # Interpolate the poses at the new timestamps
    interpolated_poses = np.zeros((new_timestamps.shape[0], poses.shape[1]))
    if new_timestamps[0] > timestamps[-1] or new_timestamps[-1] < timestamps[0]:
        print(
            "New timestamps are out of range of the original timestamps. Skipping interpolation."
        )
        return interpolated_poses
    if new_timestamps[0] < timestamps[0]:
        print("New timestamps are before the original timestamps")
    if new_timestamps[-1] > timestamps[-1]:
        print("New timestamps are after the original timestamps")
    for i in range(poses.shape[1]):
        interpolated_poses[:, i] = interp.interp1d(
            timestamps, poses[:, i], kind="cubic", assume_sorted=True
        )(new_timestamps)
    return interpolated_poses


@click.command()
@click.argument("mocap_file", type=click.Path(exists=True))
@click.argument("video_file", type=click.Path(exists=True))
@click.argument("time_difference", type=float)
def main(mocap_file: str, video_file: str, time_difference: float):
    video_name = os.path.basename(video_file).split(".")[0]
    output_file = mocap_file.replace(".pkl", f"_{video_name}.pkl")
    with open(mocap_file, "rb") as f:
        data: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = pickle.load(f)
    container = av.open(video_file)
    frame_count = 0
    video_stream = container.streams.video[0]
    print(container.metadata)
    print(video_stream.metadata)

    # Get the video framerate
    video_fps = video_stream.average_rate
    print(f"Video framerate: {video_fps} fps")

    # Example of creation_time: 2024-12-12T05:19:00.000000Z
    creation_date_time = datetime.strptime(
        video_stream.metadata["creation_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    # Example of timecode: 00:00:00:00
    video_start_system_time = timedelta(
        hours=int(video_stream.metadata["timecode"][:2]),
        minutes=int(video_stream.metadata["timecode"][3:5]),
        seconds=int(video_stream.metadata["timecode"][6:8]),
        microseconds=int(video_stream.metadata["timecode"][9:12]),
    )
    video_start_system_time = creation_date_time + video_start_system_time

    video_frame_num = container.streams.video[0].frames
    # Replace fixed 60Hz with actual framerate
    video_frame_relative_timestamps = np.arange(video_frame_num) / float(video_fps)

    video_frame_mocap_timestamps = (
        video_frame_relative_timestamps
        + video_start_system_time.timestamp()
        - time_difference
    )

    interpolated_results = {}
    for rigid_body_name, trajectory in data.items():
        original_poses = np.array([pose_xyz_wxyz for pose_xyz_wxyz, _ in trajectory])
        original_timestamps = np.array(
            [mocap_timestamp for _, mocap_timestamp in trajectory]
        )
        interpolated_poses = interpolate_mocap_poses(
            original_poses, original_timestamps, video_frame_mocap_timestamps
        )
        interpolated_results[rigid_body_name] = list(
            zip(interpolated_poses, video_frame_mocap_timestamps)
        )
    with open(output_file, "wb") as f:
        pickle.dump(interpolated_results, f)


if __name__ == "__main__":
    main()
