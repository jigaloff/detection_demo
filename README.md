## Video Streaming with Flask Example

### Description

### Usage
1. git clone https://github.com/jigaloff/detection_demo
2. cd detection_demo
3. wget https://pjreddie.com/media/files/yolov3.weights
2. Install packages

conda install -c menpo opencv
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c anaconda flask
conda install -c conda-forge flask-socketio
    For SSL
conda create --name py36 python=3.6
sudo conda install python=3.6.8
conda activate py36
conda install -c anaconda gunicorn
conda install -c conda-forge eventlet

3. Run "python main.py".
    For SSL
    openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365
    gunicorn --worker-class eventlet -w 1 --certfile cert.pem --keyfile key.pem -b 0.0.0.0:8000 main:app

4. Navigate the browser to the local webpage.

http://127.0.0.1/video - recognize on local machine without net
http://127.0.0.1/      - recognize on server
http://127.0.0.1/test  - test video from client to server and back