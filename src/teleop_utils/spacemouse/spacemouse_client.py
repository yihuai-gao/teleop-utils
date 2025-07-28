# Copyright (c) 2025 yihuai
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from robotmq import RMQClient
import numpy as np
import numpy.typing as npt
import time


class SpacemouseClient:
    def __init__(
        self,
        rmq_server_address: str = "tcp://localhost:15557",
        connection_timeout_s: float = 1.0,
    ):
        self.rmq_client = RMQClient("spacemouse_client", rmq_server_address)
        connect_start_time = time.time()
        print(f"Connecting to spacemouse server at {rmq_server_address}")
        self.get_latest_state()
        print("Spacemouse server connected")
        # while time.time() - connect_start_time < connection_timeout_s:
        #     raw_data, _ = self.rmq_client.peek_data("spacemouse_state", -1)
        #     if len(raw_data) > 0:
        #         break
        #     time.sleep(0.01)
        # else:
        #     raise RuntimeError(f"Failed to connect to spacemouse server in {connection_timeout_s} seconds. Please check if the spacemouse server is running.")

    def get_latest_state(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamp = self.rmq_client.peek_data("spacemouse_state", -1)
        spacemouse_state = np.frombuffer(raw_data[0], dtype=np.float64)
        return spacemouse_state[:6], spacemouse_state[6:]

    def get_average_state(
        self, n: int, average_buttons: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamps = self.rmq_client.peek_data("spacemouse_state", -n)
        spacemouse_states = np.array(
            [np.frombuffer(data, dtype=np.float64) for data in raw_data]
        )
        if average_buttons:
            return np.mean(spacemouse_states[:, :6], axis=0), np.mean(
                spacemouse_states[:, 6:], axis=0
            )
        else:
            return np.mean(spacemouse_states[:, :6], axis=0), spacemouse_states[0, 6:]


if __name__ == "__main__":
    spacemouse_client = SpacemouseClient()
    np.set_printoptions(precision=3, suppress=True)
    while True:
        print(spacemouse_client.get_latest_state())
        print(spacemouse_client.get_average_state(10))
        time.sleep(0.01)
