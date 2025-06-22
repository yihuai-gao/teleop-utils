<!--
 Copyright (c) 2025 yihuai
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

## Default robotmq addresses:

- iphone: `tcp://*:15555`
- mocap: `tcp://*:15556`
- spacemouse: `tcp://*:15557`
- keyboard: `tcp://*:15558`

## Install


```bash
pip install teleop-utils
# for spacemouse, you need to install spnav from source
pip install https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz
```

## Run

In one terminal, run the server and keep it running in the background. You don't need to close it once you start it.
```bash
conda activate env_1 # Choose one of your conda environments
# Run the server (choose one of them)
keyboard_server
spacemouse_server
iphone_server
mocap_server
```

In another terminal, run the test scripts in the client code. You can follow these scripts and integrate the client into your code base.

```bash
conda activate env_1 # You can use the same conda environment as the server
conda activate env_2 # Or a different conda environment (even with different python versions than the server)
# Run the corresponding client (choose one of them)
python -m teleop_utils.keyboard.keyboard_client
python -m teleop_utils.spacemouse.spacemouse_client
python -m teleop_utils.iphone.iphone_client
python -m teleop_utils.mocap.mocap_client
```

## Important Notes

- You can run the server and client in different python environments, but please make sure the numpy versions are compatible between the server and client when using `pickle.dumps` (otherwise you may get some `numpy` errors, such as `No module named numpy._core`). To solve this, you can use `robotmq.utils.serialize` to serialize nested numpy objects (list, dict, tuple, etc.).
- The keyboard server will only listen to the keyboard events in the terminal where it is running.
