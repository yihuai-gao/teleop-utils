import time
import zmq_interface as zi
from .iphone_command import TeleopData, iPhoneEvents
import pickle
import numpy as np


class iPhoneClient:
    def __init__(self, server_address: str):
        self.zmq_client = zi.ZMQClient("iPhoneClient", server_address)

    def get_latest_pose(self):
        data_bytes, timestamp = self.zmq_client.peek_data("data", "latest", 1)
        if len(data_bytes) == 0:
            return None
        iphone_pose: TeleopData = pickle.loads(data_bytes[0])
        return iphone_pose

    def get_events(self):
        event_bytes, timestamp = self.zmq_client.pop_data("events", "latest", -1)
        events: list[iPhoneEvents] = [
            pickle.loads(event_bytes[i]) for i in range(len(event_bytes))
        ]
        return events


if __name__ == "__main__":

    iphone_client = iPhoneClient("tcp://localhost:5555")
    np.set_printoptions(precision=4, suppress=True, sign=" ")
    while True:
        iphone_pose = iphone_client.get_latest_pose()
        if iphone_pose is not None:
            print(
                f"pos: {iphone_pose.position_xyz}, ori: {iphone_pose.orientation_wxyz}, gripper: {iphone_pose.gripper_speed}"
            )
        events = iphone_client.get_events()
        if events:
            print(events)
        time.sleep(0.1)
