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
from PIL import Image
from camera import VideoCamera
from darknet import Darknet
from flask import Flask, render_template, Response, request, session
from preprocess import prep_image
from io import BytesIO
from torch.autograd import Variable
from util import *

host = '127.0.0.1'
port = 5000

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

def gen(camera):
    while True:
        success, frame = camera.video.read()

        img, orig_im, dim = prep_image(frame, inp_dim)

        if CUDA:
            img = img.cuda()

        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        list(map(lambda x: write(x, orig_im), output))

        frame = camera.get_frame(frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_png(frame):

        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        img, orig_im, dim = prep_image(frame, inp_dim)

        if CUDA:
            img = img.cuda()

        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        list(map(lambda x: write(x, orig_im), output))

        frame = Image.fromarray(frame)
        return (frame) # Format: PIL image

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode, \
                           serv_addr = host, port = port)

@app.route('/test')
def test():
    return render_template('test.html', async_mode=socketio.async_mode, \
                           serv_addr = host, port = port)

@app.route('/video')
def video():
    return render_template('video.html', async_mode=socketio.async_mode)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def test_connect():
    emit('after connect',  {'data':'Lets dance'})


@socketio.on('blob')
def handle_blob(blob):
    image = Image.open(BytesIO(blob))
    image = gen_png(image)

    b = BytesIO()
    image.save(b, 'png')
    image = b.getvalue()
    emit('blob', image)

@socketio.on('test')
def handle_blob(blob):
    image = Image.open(BytesIO(blob))
    b = BytesIO()
    image.save(b, 'png')
    image = b.getvalue()
    emit('test', image)



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

    socketio.run(app, host=host)