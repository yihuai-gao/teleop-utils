# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from csv import DictWriter
import numpy as np
import time
from ..mocap_client import MocapClient
from pynput import keyboard
import numpy.typing as npt
import pickle
import os
import click


@click.command()
@click.argument("rigid_body_names", type=str, nargs=-1)
def main(rigid_body_names: list[str]):
    mocap_client = MocapClient()
    recording = False
    quit_flag = False
    results: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = {
        rigid_body_name: [] for rigid_body_name in rigid_body_names
    }
    np.set_printoptions(precision=4, suppress=True, sign=" ")

    def on_press(key):
        if key == keyboard.Key.space:
            nonlocal recording
            nonlocal results
            recording = not recording
            if recording:
                results = {rigid_body_name: [] for rigid_body_name in rigid_body_names}
                print("Recording started")
            else:
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                data_file_name = f"{'_'.join(rigid_body_names)}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
                data_file_path = os.path.join(
                    current_file_dir, "..", "data", data_file_name
                )
                print(f"Recording stopped. Saving data to {data_file_path}")
                with open(data_file_path, "wb") as f:
                    pickle.dump(results, f)
                print("Data saved")
                results = {rigid_body_name: [] for rigid_body_name in rigid_body_names}
        elif key == keyboard.Key.esc:
            nonlocal quit_flag
            quit_flag = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    last_timestamp: dict[str, float] = {
        rigid_body_name: 0 for rigid_body_name in rigid_body_names
    }
    while not quit_flag:
        if recording:
            for rigid_body_name in rigid_body_names:
                result = mocap_client.get_latest_pose_xyz_wxyz(rigid_body_name)
                if result is None:
                    continue
                pose_xyz_wxyz, mocap_timestamp = result
                if mocap_timestamp == last_timestamp[rigid_body_name]:
                    continue
                last_timestamp[rigid_body_name] = mocap_timestamp
                results[rigid_body_name].append((pose_xyz_wxyz, mocap_timestamp))
                print(
                    f"Object: {rigid_body_name}, Timestamp: {mocap_timestamp:.6f}, Position: {pose_xyz_wxyz}"
                )
        time.sleep(1 / 120)

    listener.stop()


if __name__ == "__main__":
    main()
