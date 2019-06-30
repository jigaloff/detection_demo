var video = $("#live").get()[0];
var canvas = $("#canvas");
var outcanvas = $("#outcanvas");
var ctx = canvas.get()[0].getContext('2d');

navigator.getUserMedia = navigator.getUserMedia ||
                     navigator.webkitGetUserMedia ||
                     navigator.mozGetUserMedia;

if (navigator.getUserMedia) {
   navigator.getUserMedia({ audio: true, video: { width: 320, height: 240 } },
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

//var i = 0
//socket.on('message', function(msg) {
//    console.log('Начало сообщения');
//    console.log(msg);
//    console.log('Должно быть сообщение');
////    socket.emit('message', {data: 'Test complite'});
//});

socket.on('blob', function(msg) {
//    console.log('Начало сообщения');
//    console.log(msg);
//    console.log('Должно быть сообщение');
//    socket.emit('message', {data: 'Test complite'});
});

function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], {type:mimeString});
}

timer = setInterval(
        function () {
            ctx.drawImage(video, 0, 0, 320, 240);
            var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
            var newblob = dataURItoBlob(data);

//            socket.emit('blob', {data: newblob});
        }, 250);