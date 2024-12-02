import signal
import sys
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from transforms3d import quaternions, affines, euler
import numpy as np
import zmq_interface as zi
import socket
from iphone_command import iPhoneCommand, iPhoneCommandType
import pickle


class iPhoneServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.zmq_server = zi.ZMQServer("iPhoneServer", "tcp://*:5555")
        self.zmq_server.add_topic("iphone_teleop", 10.0)

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.socketio.on("message")
        def handle_message(data):
            emit("echo", data["timestamp"])
            if "position" in data:
                new_cmd = iPhoneCommand(
                    iPhoneCommandType.SEND_DATA,
                    data["timestamp"],
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
                    0.0,
                    # data["gripper"],
                )
                self.zmq_server.put_data("iphone_teleop", pickle.dumps(new_cmd))

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
        print(f"Starting server at {address}:5000")
        self.socketio.run(self.app, host="0.0.0.0")


if __name__ == "__main__":
    iphone_server = iPhoneServer()
    iphone_server.run()
