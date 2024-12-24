# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

from typing import Dict, List, cast
import numpy as np
import sys
import time
from .natnet_client import NatNetClient
from .mocap_data import RigidBody

from functools import partial

from transforms3d import quaternions, affines
from robotmq import RMQServer
import click
import json


class MocapServer:
    def __init__(
        self,
        rigid_body_dict: Dict[int, str],
        mocap_server_ip: str,
        rmq_server_address: str = "tcp://*:15556",
        use_multicast=True,
    ):

        self.rmq_server = RMQServer("mocap_server", rmq_server_address)
        for rigid_body_name in rigid_body_dict.values():
            self.rmq_server.add_topic(rigid_body_name, 10.0)
        self.prev_receive_time = time.monotonic()

        streaming_client = NatNetClient()
        streaming_client.set_client_address("127.0.0.1")
        streaming_client.set_server_address(mocap_server_ip)
        streaming_client.set_use_multicast(use_multicast)

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        streaming_client.new_frame_listener = partial(
            self.receive_new_frame,
            rigid_body_dict=rigid_body_dict,
            rmq_server=self.rmq_server,
        )

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = streaming_client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting 1")

        time.sleep(1)
        if streaming_client.connected() is False:
            print(
                "ERROR: Could not connect properly.  Check that Motive streaming is on."
            )
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting 2")

        print("init done")

    def receive_new_frame(
        self,
        data_dict,
        rigid_body_dict: Dict[int, str],
        rmq_server: RMQServer,
    ):
        # print(f"receive_new_frame: {data_dict}")
        model_names = []
        marker_data_list = data_dict["marker_set_data"].marker_data_list
        for marker_data in marker_data_list:
            model_name = marker_data.model_name.decode("utf-8")
            if model_name != "all":
                model_names.append(model_name)

        rigid_body_list = data_dict["rigid_body_data"].rigid_body_list
        rigid_body_list = cast(List[RigidBody], rigid_body_list)
        timestamp = data_dict["timestamp"]

        for i, rigid_body in enumerate(rigid_body_list):
            if rigid_body.id_num not in rigid_body_dict:
                continue
            rigid_body_name = rigid_body_dict[rigid_body.id_num]
            name = rigid_body_name

            mocap_robot_in_world_frame = affines.compose(
                T=rigid_body.pos,
                R=quaternions.quat2mat(
                    np.array(rigid_body.rot)[[3, 0, 1, 2]]
                ),  # rigid_body.rot is xyzw, need to convert to wxyz
                Z=np.ones(3),
            )

            trans, rotm, _, _ = affines.decompose(mocap_robot_in_world_frame)
            quat_wxyz = quaternions.mat2quat(rotm)
            pose_xyz_wxyz_timestamp = np.concatenate(
                [trans, quat_wxyz, [timestamp]], dtype=np.float64
            )
            rmq_server.put_data(rigid_body_name, pose_xyz_wxyz_timestamp.tobytes())
        self.prev_receive_time = time.monotonic()


def parse_dict(ctx, param, value):
    """Parse a dictionary from a string like 'mocap_id1:object_name1,mocap_id2:object_name2'."""
    if not value:
        raise click.BadParameter(
            "Dictionary format must be 'mocap_id1:object_name1,mocap_id2:object_name2,...'."
        )
    try:
        return {int(k): v for k, v in (item.split(":") for item in value.split(","))}
    except ValueError:
        raise click.BadParameter(
            "Dictionary format must be 'mocap_id1:object_name1,mocap_id2:object_name2,...'."
        )


@click.command()
@click.argument("mocap-server-ip", type=str)
@click.argument("rigid-body-dict", callback=parse_dict)
@click.option("--rmq-server-address", type=str, default="tcp://*:15556")
def run_mocap_server(
    mocap_server_ip: str, rigid_body_dict: dict[int, str], rmq_server_address: str
):
    mocap_server = MocapServer(
        rigid_body_dict=rigid_body_dict,
        mocap_server_ip=mocap_server_ip,
        rmq_server_address=rmq_server_address,
    )
    while True:
        time.sleep(1)


if __name__ == "__main__":
    run_mocap_server()
