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

camera = nano.Camera(flip=0, width=224, height=224, fps=30)
frame = camera.read()
print('Pierwszy frame', frame)

import torch
import urllib.request
import os
import cv2
import time

import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
# from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
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


def load_model_keras(model_path=os.path.join('tf_model', 'best_model.ckpt')):
    model_loaded = tf.keras.models.Sequential()
    model_loaded.add(
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1), padding='same'))
    # model.add(LeakyReLU(alpha=0.1))
    model_loaded.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_loaded.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model_loaded.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_loaded.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model_loaded.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_loaded.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model_loaded.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model_loaded.add(tf.keras.layers.Flatten())
    model_loaded.add(tf.keras.layers.Dropout(0.5))
    model_loaded.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # model.add(tf.keras.activations.relu(alpha=0.1))
    model_loaded.add(Dropout(0.5))
    model_loaded.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model_loaded.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # model_loaded.summary()
    # print(model_loaded)
    model_loaded.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_loaded.load_weights(model_path)
    return model_loaded


def classify_image(frame):
    # image = cv2.resize(frame, (224, 224))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_arr = img_to_array(frame)
    img_arr = img_arr / 255.
    np_image = np.expand_dims(img_arr, axis=0)
    pred_value = model.predict(np_image)[0][0]
    return pred_value


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
    cv2.imwrite(f'./images/image_{i}.jpg', frame)
    return 1


if __name__ == '__main__':
    # 224
    i = 697
    print('start')
    # camera = nano.Camera(flip=0, width=224, height=224, fps=30)
    print('start camera')
    model = load_model_keras()
    print('load model')
    robot = Robot()
    print('CSI Camera ready? - ', camera.isReady())

    print('load_midas')
    midas, transform, device = load_model_midas()
    # while camera.isReady():
    while True:
        # try:
        print('Next frame ', end='')
        frame = camera.read()
        if frame is None:
            print('Frame is NONE', camera.isReady())
            camera = nano.Camera(flip=0, width=224, height=224, fps=30)
            continue
        # i += save_frame(frame)
        depth_image = predict_midas(frame)
        prob_free = classify_image(depth_image)
        is_free = prob_free > 0.3
        print(f': {"Free" if is_free else "Block"}: {prob_free * 100:.2f}%')
        move_robot(prob_free)

    # except Exception as e:
    #     print("ERROR: ", e)

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