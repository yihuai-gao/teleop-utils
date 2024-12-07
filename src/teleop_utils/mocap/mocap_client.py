import numpy as np
import robotmq


class MocapClient:
    def __init__(self, server_address: str = "tcp://localhost:5556"):
        self.rmq_client = robotmq.RMQClient("mocap_client", server_address)

    def get_latest_pose_xyz_wxyz(self, rigid_body_name: str):
        data_bytes, timestamp = self.rmq_client.peek_data(rigid_body_name, "latest", 1)
        if len(data_bytes) == 0:
            return None
        pose_xyz_wxyz_timestamp = np.frombuffer(data_bytes[0], dtype=np.float64)
        return pose_xyz_wxyz_timestamp[:7], pose_xyz_wxyz_timestamp[7]
