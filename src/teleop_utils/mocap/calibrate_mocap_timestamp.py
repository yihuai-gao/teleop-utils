import qrcode

import numpy as np
import time
from .mocap_client import MocapClient
import matplotlib.pyplot as plt

def main():
    mocap_client = MocapClient()
    fps = 60
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    while True:
        start_time = time.time()
        results = mocap_client.get_latest_pose_xyz_wxyz("umi_gripper")
        if results is None:
            continue
        pose_xyz_wxyz, mocap_timestamp = results
        qr.clear()
        qr.add_data(f"{mocap_timestamp:.6f}")
        qr.make(fit=True)
        img = qr.make_image(fill="black", back_color="white")
        ax.imshow(img)
        while time.time() - start_time < 1 / fps:
            pass



if __name__ == "__main__":
    main()
