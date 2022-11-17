# scp ./move_jetbot_17_11.py jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance
# scp -r jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/images ./images
# scp -r ./images/dataset jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/dataset
# scp -r ./tf_model jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance
# scp -r ./jetbot jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/jetbot
# jupyter notebook --no-browser --port=8888 --ip 0.0.0.0

# FIX na debilne błędy:
# rm -rf ~/.cache/gstreamer-1.0
# source ~/.bashrc
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1


# %pip install timm
import nanocamera as nano

camera = nano.Camera(flip=0, width=640, height=640, fps=10)
frame = camera.read()
print('Pierwszy frame', frame)

import torch
import urllib.request
import os
import cv2
import time
import numpy as np
from jetbot import Robot


def load_model_midas():
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform, device


def predict_midas(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def move_robot(prob_free, speed=0.3, sleep_time=0.1):
    if prob_free > 0.3:
        robot.forward(speed)
    else:
        robot.backward(speed)
        time.sleep(sleep_time)
        robot.left(speed)

    time.sleep(sleep_time)
    robot.stop()


def save_frame(frame):
    cv2.imwrite(f'./images/image_{i}.jpg', frame[120:160])
    return 1


def average(depth_image):
    avg = np.max(depth_image)
    # avg1 = np.max(depth_image[0:40, :])
    # depth_image.shape

    avg_left = np.average(depth_image[200:400, 50:200])
    avg_mid = np.average(depth_image[200:400, 200:450])
    avg_right = np.average(depth_image[200:400, 450:600])

    # avg5 = np.max(depth_image[160:200, :])
    print('average', avg, ' = ', avg_left, avg_mid, avg_right, depth_image.shape)


if __name__ == '__main__':
    i = 697
    print('start')
    robot = Robot()
    print('CSI Camera ready? - ', camera.isReady())

    print('load_midas')
    midas, transform, device = load_model_midas()
    while True:
        print('Next frame ', end='')
        frame = camera.read()
        if frame is None:
            print('Frame is NONE', camera.isReady())
            camera = nano.Camera(flip=0, width=640, height=640, fps=30)
            continue
        # i += save_frame(frame)
        depth_image = predict_midas(frame)
        average(depth_image)
        time.sleep(2)
        # move_robot(is_free)

    print('Robot.stop()')
    robot.stop()
    print('robot stopped')
    camera.release()
    del camera
    print('end')

# (gst-plugin-scanner:17764): GStreamer-WARNING **: 12:24:08.588: Failed to load plugin '/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvcompositor.so': libgstbadvideo-1.0.so.
# 0: cannot open shared object file: No such file or directory
# (Argus) Error Timeout:  (propagating from src/rpc/socket/client/SocketClientDispatch.cpp, function openSocketConnection(), line 219)
# (Argus) Error Timeout: Cannot create camera provider (in src/rpc/socket/client/SocketClientDispatch.cpp, function createCameraProvider(), line 106)
