## Video Streaming with Flask Example

### Description
Modified to support streaming out with webcams, and not just raw JPEGs.


### Usage
1. Download 'wget https://pjreddie.com/media/files/yolov3.weights'
conda create --name py36 python=3.6
sudo conda install python=3.6.8
conda activate py36
conda install -c anaconda gunicorn
conda install -c conda-forge eventlet
conda install -c menpo opencv
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c anaconda flask
conda install -c conda-forge flask-socketio

openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 \
  -keyout certs/key.pem -out certs/cert.pem
gunicorn --worker-class eventlet -w 1 --certfile cert.pem --keyfile key.pem -b 0.0.0.0:8000 main:app
2. Install Python dependencies: cv2, flask, flask_socketio and other with requirements.txt
3. Run "python main.py".
4. Navigate the browser to the local webpage.

###
http://localhost/video - recognize on local machine without net
http://lacalhost/      - recognize on server
http://lacalhost/test  - test video from client to server and back