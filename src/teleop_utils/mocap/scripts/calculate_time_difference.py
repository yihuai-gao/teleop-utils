# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

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
from datetime import datetime, timedelta


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
def main(video_path: str):

    container = av.open(video_path)
    frame_count = 0
    video_stream = container.streams.video[0]
    print(video_stream.metadata)
    # Example of creation_time: 2024-12-12T05:19:00.000000Z
    creation_date_time = datetime.strptime(
        video_stream.metadata["creation_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    # Example of timecode: 00:00:00:00
    video_start_system_time = timedelta(
        hours=int(video_stream.metadata["timecode"][:2]),
        minutes=int(video_stream.metadata["timecode"][3:5]),
        seconds=int(video_stream.metadata["timecode"][6:8]),
        microseconds=int(video_stream.metadata["timecode"][9:12]),
    )
    video_start_system_time = creation_date_time + video_start_system_time

    time_differences_s: list[float] = []
    try:
        for frame in container.decode(video=0):  # Decode video stream (index 0)
            frame_count += 1
            # Convert the frame to an image format suitable for decoding
            img = frame.to_image()  # PIL Image
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Decode QR codes in the frame
            qrcodes = decode(img_cv)

            for qrcode in qrcodes:
                # Extract and print QR code data
                qrcode_data: str = qrcode.data.decode("utf-8")
                side_data = frame.side_data
                frame_system_time = video_start_system_time + timedelta(
                    seconds=frame.time
                )
                time_difference = frame_system_time.timestamp() - float(qrcode_data)
                time_differences_s.append(time_difference)
                print(
                    f"Frame {frame_count}: QR Code Data: {qrcode_data}, frame system time: {frame_system_time.strftime('%H:%M:%S.%f')}, time difference: {time_difference}"
                )
    except KeyboardInterrupt:
        print(f"Interrupted by user")

    print(
        f"Average time difference (camera timestamp - mocap timestamp): {np.mean(time_differences_s)}"
    )


if __name__ == "__main__":
    main()
