## Default robotmq addresses:

- iphone: `tcp://*:15555`
- mocap: `tcp://*:15556`
- spacemouse: `tcp://*:15557`
- keyboard: `tcp://*:15558`

## Install

```bash
# If you want to run all the example scripts, you can create a conda environment with all the dependencies
conda env create -f conda_environment.yaml
# Otherwise, only install some required dependencies for server and client (into your current environment)
conda install cmake spdlog cppzmq zeromq boost pybind11

# Install robot-message-queue
git clone https://github.com/yihuai-gao/robot-message-queue.git ../robot-message-queue
pip install -e ../robot-message-queue

# Install teleop-utils
pip install -e .
```

## Run

In one terminal, run the server and keep it running in the background. You don't need to close it once you start it.
```bash
conda activate teleop-utils # Or some other environment
# Run the server (choose one of them)
keyboard_server
spacemouse_server
iphone_server
mocap_server
```

In another terminal, run the test scripts in the client code. You can follow these scripts and integrate the client into your code base.

```bash
conda activate teleop-utils # Or another environment (even with different python versions than the server)
# Run the client (choose one of them)
python -m teleop_utils.keyboard.keyboard_client
python -m teleop_utils.spacemouse.spacemouse_client
python -m teleop_utils.iphone.iphone_client
python -m teleop_utils.mocap.mocap_client
```

## Important Notes

- You can run the server and client in different python environments, but please make sure the numpy versions are compatible between the server and client when using `pickle.dumps` (otherwise you may get some `numpy` errors, such as `No module named numpy._core`). This should be fixed in the robotmq package, but just in case there's still something wrong.
- The keyboard server will only listen to the keyboard events in the terminal where it is running.
