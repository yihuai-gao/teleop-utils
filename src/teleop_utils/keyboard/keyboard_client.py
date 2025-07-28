# Copyright (c) 2025 yihuai
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import click
from robotmq import RMQClient
import time


class KeyboardClient:
    def __init__(self, rmq_server_address: str = "tcp://localhost:15558"):
        self.rmq_client: RMQClient = RMQClient("keyboard_client", rmq_server_address)
        print(f"Connecting to keyboard server at {rmq_server_address}")
        self.rmq_client.pop_data("keyboard", 0)  # pop all previous data
        print("Keyboard server connected")

    def get_keys(self):
        keys, timestamps = self.rmq_client.pop_data("keyboard", 0)
        return [key.decode("utf-8") for key in keys]


if __name__ == "__main__":
    client = KeyboardClient()
    while True:
        print(client.get_keys())
        time.sleep(0.1)
