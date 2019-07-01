## Video Streaming with Flask Example

### Description

### Usage
1. Download 'wget https://pjreddie.com/media/files/yolov3.weights'

2. Install packages
# conda create --name py36 python=3.6 # Only for SSL
# sudo conda install python=3.6.8 # Only for SSL
# conda activate py36 # Only for SSL
conda install -c anaconda gunicorn #For ssl
conda install -c conda-forge eventlet #For ssl
conda install -c menpo opencv
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c anaconda flask
conda install -c conda-forge flask-socketio

3. Run "python main.py".
   ## For SSL #  openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365
   ## For SSL #  gunicorn --worker-class eventlet -w 1 --certfile cert.pem --keyfile key.pem -b 0.0.0.0:8000 main:app

4. Navigate the browser to the local webpage.
###
http://127.0.0.1/video - recognize on local machine without net
http://127.0.0.1/      - recognize on server
http://127.0.0.1/test  - test video from client to server and back