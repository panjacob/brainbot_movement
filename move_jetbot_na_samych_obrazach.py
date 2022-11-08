# scp ./move_jetbot.py jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance
# scp -r jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/images ./images
# scp -r ./images/dataset jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/dataset
# scp -r ./tf_model jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance
# scp -r ./jetbot jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/jetbot
# jupyter notebook --no-browser --port=8888 --ip 0.0.0.0
# scp ./udp_client.py jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance


# FIX na debilne błędy:
# rm -rf ~/.cache/gstreamer-1.0
# source ~/.bashrc
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
import os
import cv2
import time
import nanocamera as nano
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
from jetbot import Robot




def load_model(model_path=os.path.join('tf_model', 'best_model.ckpt')):
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


# def preprocess(camera_value):
#     global device, normalize
#     x = camera_value
#     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     x = x.transpose((2, 0, 1))
#     x = torch.from_numpy(x).float()
#     x = normalize(x)
#     x = x.to(device)
#     x = x[None, ...]
#     return x


def classify_image(frame):
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_arr = img_to_array(image)
    img_arr = img_arr / 255.
    np_image = np.expand_dims(img_arr, axis=0)
    pred_value = model.predict(np_image)[0][0]
    return pred_value


def move_robot(prob_free, speed=0.3, sleep_time=0.1):
    if prob_free > 0.5:
        robot.forward(speed)
    else:
        robot.backward(speed)
        time.sleep(sleep_time)
        robot.left(speed)

    # time.sleep(sleep_time)
    # robot.stop()


def save_frame(frame):
    cv2.imwrite(f'./images/image_{i}.jpg', frame)
    return 1


if __name__ == '__main__':
    # 224
    i = 697
    print('start')
    camera = nano.Camera(flip=0, width=224, height=224, fps=30)
    print('start camera')
    model = load_model()
    print('load model')
    robot = Robot()
    print('CSI Camera ready? - ', camera.isReady())
    while camera.isReady():
        try:
            print('Next frame', end='')
            frame = camera.read()
            i += save_frame(frame)
            prob_free = classify_image(frame)
            is_free = prob_free > 0.9
            print(f': {"Free" if is_free else "Block"}: {prob_free * 100:.2f}%')
            move_robot(prob_free)

        except:
            break

    print('Robot.stop()')
    robot.stop()
    print('robot stopped')
    camera.release()
    del camera
    print('end')
