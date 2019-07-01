var video = $("#live").get()[0];
var canvas = $("#canvas");
var outcanvas = $("#outcanvas");
var ctx = canvas.get()[0].getContext('2d');

navigator.getUserMedia = navigator.getUserMedia ||
                     navigator.webkitGetUserMedia ||
                     navigator.mozGetUserMedia;

if (navigator.getUserMedia) {
   navigator.getUserMedia({ audio: false, video: { width: 320, height: 240 } },
      function(stream) {
         var video = document.querySelector('video');
         video.srcObject = stream;
         video.onloadedmetadata = function(e) {
           video.play();
         };
      },
      function(err) {
         console.log("The following error occurred: " + err.name);
      }
   );
} else {
   console.log("getUserMedia not supported");
};

var socket = io.connect('http://127.0.0.1:5000');

socket.on('connect', function() {
    console.log("Openened connection to websocket");
    socket.emit('message', {data: 'I\'m connected!'});
});

let urlObject;

function blob2canvas(blob){
            var blobData = blob;
            var url = window.URL || window.webkitURL;
            var src = url.createObjectURL(blob);
            $('#image').attr("src", src);
}
var blobs

socket.on('blob', function(blob) {
    blob = new Blob([blob], {type: "image/png"});
    blob2canvas(blob)
});

timer = setInterval(
        function () {
            ctx.drawImage(video, 0, 0, 320, 240);
            var canvas = document.getElementById('canvas');
            canvas.toBlob(function(blob) {
              socket.emit('blob', blob);
//              console.log(blob);
            });
        }, 250);