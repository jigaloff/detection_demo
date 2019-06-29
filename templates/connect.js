//$(document).ready(function(){
////    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
//    var socket = io.connect('http://localhost:8000');
////    socket.on('my response', function(msg) {
//        $('#log').append('<p>Received: ' + msg.data + '</p>');
//    });
//    $('form#emit').submit(function(event) {
//        socket.emit('my event', {data: $('#emit_data').val()});
//        return false;
//    });
//    $('form#broadcast').submit(function(event) {
//        socket.emit('my broadcast event', {data: $('#broadcast_data').val()});
//        return false;
//    });
//
//    startbutton.addEventListener('click', function(ev){
////      takepicture();
////      ev.preventDefault();
//        alert("sometext");
//    }, false);


$( init );

function init() {
  $('#sendbutton').bind( 'click', sayHello );
}

function sayHello() {
//  alert( "Всем - привет!" );
  ws.send("Привет");
}

//});

