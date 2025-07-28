# Copyright (c) 2025 yihuai
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import click
import numpy as np
import robotmq
import time
import numpy.typing as npt


class MocapClient:
    def __init__(self, server_address: str = "tcp://localhost:15556"):
        self.rmq_client = robotmq.RMQClient("mocap_client", server_address)

    def get_latest_pose_xyz_wxyz(self, rigid_body_name: str):
        data_bytes, timestamps = self.rmq_client.peek_data(rigid_body_name, -1)
        if len(data_bytes) == 0:
            return None
        pose_xyz_wxyz_timestamp = np.frombuffer(data_bytes[0], dtype=np.float64)
        return pose_xyz_wxyz_timestamp[:7], pose_xyz_wxyz_timestamp[7]

    def get_average_pose_xyz_wxyz(
        self, rigid_body_name: str, n: int = 10
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamps = self.rmq_client.peek_data(rigid_body_name, -n)
        poses_xyz_wxyz = np.array(
            [np.frombuffer(data, dtype=np.float64) for data in raw_data]
        )
        return np.mean(poses_xyz_wxyz[:, :7], axis=0), np.mean(
            poses_xyz_wxyz[:, 7:], axis=0
        )


@click.command()
@click.argument("rigid-body-name")
@click.option("--server-address", default="tcp://localhost:15556")
def main(rigid_body_name: str, server_address: str):
    mocap_client = MocapClient(server_address)
    np.set_printoptions(precision=3, suppress=True)
    while True:
        print(mocap_client.get_latest_pose_xyz_wxyz(rigid_body_name))
        time.sleep(0.01)


if __name__ == "__main__":
    main()
