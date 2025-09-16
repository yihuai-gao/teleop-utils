# Copyright (c) 2025 yihuai
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
Spacemouse server adapted from https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/spacemouse_shared_memory.py by Cheng Chi
"""

import numpy as np
import numpy.typing as npt

import robotmq
import click
import time


class SpacemouseServer:
    def __init__(
        self,
        rmq_server_address: str = "tcp://*:15557",
        frequency: float = 200,
        deadzone: float = 0.0,
        n_buttons: int = 2,
        max_value: int = 500,
    ):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0, the rest will be scaled to [0,1]. Recommended value: 0.1 for wired version and 0.0 for wireless version
        max_value: int, maximum value of the space mouse, 300 for the wired version and 500 for the wireless version

        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """

        # copied variables
        self.frequency = frequency
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        self.max_value = max_value
        self.tx_zup_spnav = np.array(
            [[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        self.rmq_server = robotmq.RMQServer("spacemouse_server", rmq_server_address)
        self.rmq_server.add_topic("spacemouse_state", 10.0)

    def get_motion_state_transformed(self, state: npt.NDArray[np.float64]):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def run(self):

        try:
            try:
                from spnav import (
                    SpnavButtonEvent,
                    SpnavMotionEvent,
                    spnav_close,
                    spnav_open,
                    spnav_poll_event,
                    SpnavConnectionException,
                )
            except ImportError as e:
                raise ImportError(
                    "Package `spnav` not found. Please install it with \n\tpip install https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz"
                )

        except AttributeError as e:
            raise ImportError(
                "Package `spnav` version is incompatible. Please install it with \n\tpip install --force-reinstall https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz"
            )

        try:
            spnav_open()
        except SpnavConnectionException as e:
            raise RuntimeError(
                """Failed to connect to the spacemouse. 
Please check if the spacemouse is connected and the permissions are set correctly.
To enable the spacemouse connection service, please run the following commands:
    sudo apt install libspnav-dev spacenavd
    sudo systemctl enable spacenavd.service
    sudo systemctl start spacenavd.service
            """
            )

        try:
            motion_event = np.zeros((6,), dtype=np.float64)
            button_state = np.zeros((self.n_buttons,), dtype=np.float64)
            prev_time = time.time()
            while True:
                sleep_time = prev_time + 1 / self.frequency - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                prev_time += 1 / self.frequency
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    motion_event[:3] = event.translation
                    motion_event[3:6] = event.rotation
                    motion_event /= float(self.max_value)
                    # motion_event[6] = event.period
                elif isinstance(event, SpnavButtonEvent):
                    button_state[event.bnum] = float(event.press)

                state = self.get_motion_state_transformed(motion_event)
                positive_idx = state >= self.deadzone
                negative_idx = state <= -self.deadzone
                state[positive_idx] = (state[positive_idx] - self.deadzone) / (
                    1 - self.deadzone
                )
                state[negative_idx] = (state[negative_idx] + self.deadzone) / (
                    1 - self.deadzone
                )
                state = np.concatenate([state, button_state])

                self.rmq_server.put_data("spacemouse_state", state.tobytes())

        finally:
            spnav_close()


@click.command()
@click.option("--rmq-server-address", type=str, default="tcp://*:15557")
@click.option("--frequency", type=float, default=200)
@click.option("--deadzone", type=float, default=0.0)
@click.option("--n-buttons", type=int, default=2)
def run_spacemouse_server(
    rmq_server_address: str, frequency: float, deadzone: float, n_buttons: int
):
    server = SpacemouseServer(
        rmq_server_address=rmq_server_address,
        frequency=frequency,
        deadzone=deadzone,
        n_buttons=n_buttons,
    )
    server.run()


if __name__ == "__main__":
    run_spacemouse_server()
