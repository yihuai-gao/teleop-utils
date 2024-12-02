import zmq_interface as zi
from iphone_command import iPhoneCommand
import pickle


class iPhoneClient:
    def __init__(self, server_address: str):
        self.zmq_client = zi.ZMQClient("iPhoneClient", server_address)

    def get_latest_command(self):
        data_bytes, timestamp = self.zmq_client.request_latest("iphone_teleop")
        iphone_command: iPhoneCommand = pickle.loads(data_bytes)
        return iphone_command


if __name__ == "__main__":

    while True:
        iphone_client = iPhoneClient("tcp://localhost:5555")
        iphone_command = iphone_client.get_latest_command()
        print(iphone_command.position_xyz)
        print(iphone_command.orientation_wxyz)
