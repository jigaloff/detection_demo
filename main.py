#!/usr/bin/env python

from __future__ import division

import argparse
import pickle as pkl
# import pandas as pd
import random

# import numpy as np
import cv2
# import time
# import torch.nn as nn
import torch
from camera import VideoCamera
from darknet import Darknet
from flask import Flask, render_template, Response, request, session
from preprocess import prep_image
from io import BytesIO
from torch.autograd import Variable
from util import *


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

# @app.route('/test')
# def test():
#     print('Start test')
#     send('test', {'msg': 'hello!'}, broadcast=True)
#     # send({'Data':'Test Server to Client'}, room=current_user.id)
#     print('Send Message to client complite')
#     return ""

@socketio.on('connect')
def test_connect():
    emit('after connect',  {'data':'Lets dance'})

@socketio.on('message')
def handle_message(message):
    # emit('message', {'data':'Lets dance'})
    # print(message)
    pass

import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

@socketio.on('blob')
def handle_blob(blob):
    # print(blob)
    emit('blob', blob)
    image = Image.open(BytesIO(blob))
    print(image.size)
    print(type(image))

    # blob = np.genfromtxt(BytesIO(blob))
    # blob = np.array(blob)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def gen(camera):
    i = 0
    # videofile = 'video.avi'
    # classes = load_classes('data/coco.names')
    # colors = pkl.load(open("pallete", "rb"))

    while True:
        success, frame = camera.video.read()
        if i == 1:
            img, orig_im, dim = prep_image(frame, inp_dim)

            if CUDA:
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            list(map(lambda x: write(x, orig_im), output))
            i = 0
        i += 1
        frame = camera.get_frame(frame)

        return (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    start = 0
    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    # app.run(host='127.0.0.1', debug=True)
    # app.run(host='0.0.0.0', debug=True)
    # socketio.run(app, host='0.0.0.0')
    socketio.run(app, host='127.0.0.1')