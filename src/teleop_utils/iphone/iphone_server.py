import signal
import sys
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from transforms3d import quaternions, affines, euler
import numpy as np
import robotmq
import socket

from .iphone_command import iPhoneEvents, TeleopData
import pickle


class iPhoneServer:
    def __init__(self, server_address: str = "tcp://*:5555", iphone_port: int = 5000):
        self.app: Flask = Flask(__name__)
        self.socketio: SocketIO = SocketIO(self.app)
        self.rmq_server: robotmq.RMQServer = robotmq.RMQServer(
            "iPhoneServer", server_address
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


if __name__ == "__main__":
    iphone_server = iPhoneServer()
    iphone_server.run()
