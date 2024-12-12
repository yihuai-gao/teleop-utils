"""
This script calculates the time difference between the mocap timestamp and the video recorder timestamp.
For example, if you are using GoPro to shoot the video of the mocap qr code, this script will provide
the time difference between the mocap system and the video recorder system based on the video creation time.
"""

import click
import time
import numpy as np
import numpy.typing as npt
import av
import cv2
from pyzbar.pyzbar import decode

@click.command()
@click.argument("video_path", type=click.Path(exists=True))
def main(video_path: str):
    
    container = av.open(video_path)
    frame_count = 0
    print(container.metadata)

    for frame in container.decode(video=0):  # Decode video stream (index 0)
        frame_count += 1
        # Convert the frame to an image format suitable for decoding
        img = frame.to_image()  # PIL Image
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Decode QR codes in the frame
        qrcodes = decode(img_cv)

        for qrcode in qrcodes:
            # Extract and print QR code data
            qrcode_data = qrcode.data.decode("utf-8")
            side_data = frame.side_data
            for side_data_item in side_data:
                print(side_data_item)
            print(f"Frame {frame_count}: QR Code Data: {qrcode_data}, frame time: {frame.time}")


if __name__ == "__main__":
    main()
