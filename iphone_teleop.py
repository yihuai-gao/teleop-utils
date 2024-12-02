import signal
import sys
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from transforms3d import quaternions, affines,euler
import numpy as np
def signal_handler(sig, frame):
    sys.exit()

# Set the signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)

app = Flask(__name__)
socketio = SocketIO(app)

WINDOW_SIZE = 5
prev_eef_commands = np.zeros([WINDOW_SIZE, 6])    

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    emit('echo', data['timestamp'])
    if 'position' in data:
        position = data['position']
        orientation = data['orientation']
        
        phone_pose = affines.compose(
            T=[position['x'], position['y'], position['z']],
            R=quaternions.quat2mat([orientation['w'], orientation['x'], orientation['y'], orientation['z']]),
            Z = [1, 1, 1]
        )
        
        transform_mat = np.array([
            [0, 0, 1, 0], 
            [1, 0, 0, 0], 
            [0, 1, 0, 0],
            [0, 0, 0, 1]]
            )
        

        transformed_phone_mat = np.dot(transform_mat, phone_pose)
        position["x"] = transformed_phone_mat[0, 3]
        position["y"] = transformed_phone_mat[1, 3]
        position["z"] = transformed_phone_mat[2, 3]
        transformed_quat = quaternions.mat2quat(transformed_phone_mat[:3, :3])
        orientation["w"] = transformed_quat[0]
        orientation["x"] = transformed_quat[1]
        orientation["y"] = transformed_quat[2]
        orientation["z"] = transformed_quat[3]

        r, p, y = euler.quat2euler([orientation['w'], orientation['x'], orientation['y'], orientation['z']])
        r -= np.pi/2
        y -= np.pi/2
        r, p = p, r
        r = -r
        pose_euler = np.array([r, p, y])
        r, p, y = np.mod(pose_euler + np.pi, 2*np.pi) - np.pi

        pose6d = np.array([position["x"], position["y"], position["z"], r, p, y])
        global prev_eef_commands
        prev_eef_commands = np.vstack([prev_eef_commands[1:], pose6d])
        avg_eef_command = np.mean(prev_eef_commands, axis=0)



np.set_printoptions(precision=3, suppress=True)

socketio.run(app, host='0.0.0.0', port=5000)
