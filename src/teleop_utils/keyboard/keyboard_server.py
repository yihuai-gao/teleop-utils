# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import click
import robotmq
import curses
import time


class KeyboardServer:
    def __init__(self, stdscr: curses.window, rmq_server_address: str = "tcp://*:15558"):
        self.stdscr = stdscr
        self.rmq_server = robotmq.RMQServer("keyboard_server", rmq_server_address)
        self.rmq_server.add_topic("keyboard", 30.0)

    def run(self):
        curses.curs_set(0)
        self.stdscr.keypad(True)
        self.stdscr.timeout(100)

        key_press_cnt = 0
        try:
            while True:
                key = self.stdscr.getch()
                if key == -1:
                    continue
                key_name = curses.keyname(key).decode("utf-8")
                key_press_cnt += 1
                print(f"Key pressed {key_press_cnt}: {key_name}")
                self.rmq_server.put_data("keyboard", key_name.encode("utf-8"))
        except KeyboardInterrupt:
            print("Keyboard server ended by user")


@click.command()
@click.option("--rmq-server-address", default="tcp://*:15558")
def run_keyboard_server(rmq_server_address: str):
    server = curses.wrapper(KeyboardServer, rmq_server_address)
    server.run()


if __name__ == "__main__":
    run_keyboard_server()
