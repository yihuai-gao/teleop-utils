# Copyright (c) 2025 yihuai
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import robotmq
import socket

from .iphone_command import iPhoneEvents, TeleopData
import pickle
import click

class iPhoneServer:
    def __init__(self, rmq_server_address: str, iphone_port: int):
        self.app: Flask = Flask(__name__)
        self.socketio: SocketIO = SocketIO(self.app)
        self.rmq_server: robotmq.RMQServer = robotmq.RMQServer(
            "iPhoneServer", rmq_server_address
        )
        self.rmq_server.add_topic("data", 10.0)
        self.rmq_server.add_topic("events", 10.0)
        self.iphone_port = iphone_port

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.socketio.on("message")
        def handle_message(data):
            emit("echo", data["timestamp"])
            if "position" in data:
                new_cmd = TeleopData(
                    data["timestamp"],
                    data["xr_timestamp"],
                    np.array(
                        [
                            data["position"]["x"],
                            data["position"]["y"],
                            data["position"]["z"],
                        ]
                    ),
                    np.array(
                        [
                            data["orientation"]["w"],
                            data["orientation"]["x"],
                            data["orientation"]["y"],
                            data["orientation"]["z"],
                        ]
                    ),
                    data["gripper_speed"],
                )
                self.rmq_server.put_data("data", pickle.dumps(new_cmd))
            if "event" in data:
                event_str: str = data["event"]
                new_event = iPhoneEvents[event_str.upper()]
                self.rmq_server.put_data("events", pickle.dumps(new_event))

    def run(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 1))
            address = s.getsockname()[0]
        except Exception:
            address = "127.0.0.1"
        finally:
            s.close()
        print(f"Starting server at {address}:{self.iphone_port}")
        self.socketio.run(self.app, host="0.0.0.0")


@click.command()
@click.option("--iphone-port", type=int, default=5000)
@click.option("--rmq-server-address", type=str, default="tcp://*:15555")
def run_iphone_server(iphone_port: int, rmq_server_address: str):
    iphone_server = iPhoneServer(iphone_port=iphone_port, rmq_server_address=rmq_server_address)
    iphone_server.run()

if __name__ == "__main__":
    run_iphone_server()
