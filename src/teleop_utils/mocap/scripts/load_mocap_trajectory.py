# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import numpy.typing as npt
import pickle
import click


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
def main(data_file: str):
    with open(data_file, "rb") as f:
        data: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = pickle.load(f)
    # print(len(data), data[0])
    print(len(data))
    if len(data) == 0:
        print("No data found")
        return
    for rigid_body_name, trajectory in data.items():
        for pose_xyz_wxyz, mocap_timestamp in trajectory[:100]:
            print(rigid_body_name, pose_xyz_wxyz, mocap_timestamp)


if __name__ == "__main__":
    main()
