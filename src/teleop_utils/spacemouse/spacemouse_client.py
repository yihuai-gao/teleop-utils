from robotmq import RMQClient
import numpy as np
import numpy.typing as npt
import time


class SpacemouseClient:
    def __init__(self, rmq_server_address: str = "tcp://localhost:5557"):
        self.rmq_client = RMQClient("spacemouse_client", rmq_server_address)

    def get_latest_state(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamp = self.rmq_client.peek_data("spacemouse_state", "latest", 1)
        spacemouse_state = np.frombuffer(raw_data[0], dtype=np.float64)
        return spacemouse_state[:6], spacemouse_state[6:]

    def get_average_state(
        self, n: int = 10
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_data, timestamps = self.rmq_client.peek_data(
            "spacemouse_state", "latest", n
        )
        spacemouse_states = np.array(
            [np.frombuffer(data, dtype=np.float64) for data in raw_data]
        )
        return np.mean(spacemouse_states[:, :6], axis=0), np.mean(
            spacemouse_states[:, 6:], axis=0
        )


if __name__ == "__main__":
    spacemouse_client = SpacemouseClient()
    np.set_printoptions(precision=3, suppress=True)
    while True:
        print(spacemouse_client.get_latest_state())
        time.sleep(0.01)
