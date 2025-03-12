# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import qrcode
import numpy as np
import time
from ..mocap_client import MocapClient
import pygame
import click
from PIL import Image


@click.command()
@click.argument("rigid_body_name", type=str)
def main(rigid_body_name: str):
    # Initialize Mocap Client
    mocap_client = MocapClient()
    fps = 30

    # Pygame initialization
    pygame.init()
    screen_size = (800, 800)  # Adjust screen size as needed
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("QR Code Display")

    clock = pygame.time.Clock()

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    running = True
    last_time = time.time()
    while running:
        start_time = time.time()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fetch mocap data
        results = mocap_client.get_latest_pose_xyz_wxyz(rigid_body_name)
        if results is None:
            continue

        pose_xyz_wxyz, mocap_timestamp = results

        # Generate QR code
        qr.clear()
        qr.add_data(f"{mocap_timestamp:.6f}")
        qr.make(fit=True)
        img = qr.make_image(fill="black", back_color="white")

        # Convert PIL image to pygame surface
        img = img.convert("RGB")  # Ensure compatibility
        mode = img.mode
        size = img.size
        data = img.tobytes()
        qr_surface = pygame.image.fromstring(data, size, "RGB")

        # Scale QR code to fit screen
        qr_surface = pygame.transform.scale(qr_surface, screen_size)

        # Render
        screen.fill((255, 255, 255))  # Clear screen with white background
        screen.blit(qr_surface, (0, 0))
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
