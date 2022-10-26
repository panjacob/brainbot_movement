# scp ./move_jetbot.py jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance
# scp -r jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/images ./images
# scp -r ./images/dataset jetson@192.168.0.240:/home/jetson/Documents/jetbot-master/notebooks/collision_avoidance/dataset
# jupyter notebook --no-browser --port=8888 --ip 0.0.0.0

# source ~/.bashrc
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

import cv2
# from nanocamera.NanoCam import Camera
import nanocamera as nano
import torch
import torchvision
import numpy as np
# try:
#     from jetbot import Robot
# except:
#     pass
import torch.nn.functional as F
import time


def load_model(model_path='best_model_1.pth'):
    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load(model_path))
    # device = torch.device('cuda')
    model = model.to(device)
    return model


mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)


def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


def classify_image(frame):
    x = preprocess(frame)
    y = model(x)
    print('y', y)
    y = F.softmax(y, dim=1)
    prob_blocked = float(y.flatten()[0])
    return prob_blocked


# def move_robot(prob_blocked, speed=1, sleep_time=0.001):
#     if prob_blocked < 0.5:
#         robot.forward(speed)
#     else:
#         robot.left(speed)
#     time.sleep(sleep_time)




def save_frame(frame):
    cv2.imwrite(f'./images/image_{i}.jpg', frame)
    return i + 1


if __name__ == '__main__':
    # 224
    i = 0
    device = torch.device('cuda')
    print('start')
    camera = nano.Camera(flip=0, width=224, height=224, fps=1)
    print('start camera')
    model = load_model()
    print('load model')
    # robot = Robot()
    print('CSI Camera ready? - ', camera.isReady())
    while camera.isReady():
        try:
            print('read frame')
            frame = camera.read()
            i = save_frame(frame)
            prob_blocked = classify_image(frame)
            print('prob_blocked:', prob_blocked)
            # cv2.imshow("Video Frame", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break

    # print('Robot.stop()')
    # robot.stop()
    # print('robot stopped')
    camera.release()
    del camera
    print('end')
