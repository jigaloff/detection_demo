## Video Streaming with Flask Example

### Description
Modified to support streaming out with webcams, and not just raw JPEGs.

### Usage
1. Download 'wget https://pjreddie.com/media/files/yolov3.weights'
2. Install Python dependencies: cv2, flask and other with requirements.txt
3. Run "python main.py".
4. Navigate the browser to the local webpage.

###
http://localhost/video - recognize on local machine without net
http://lacalhost/      - recognize on server
http://lacalhost/test  - test video from client to server and back