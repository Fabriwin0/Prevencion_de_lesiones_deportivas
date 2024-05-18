// script.js

function capturarImagen() {
    // Obtener el elemento de video
    var video = document.createElement('video');
    var contenedorImagen = document.getElementById('contenedor-imagen');

    // Solicitar acceso a la cámara
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            // Mostrar el video en el contenedor
            contenedorImagen.appendChild(video);
            video.srcObject = stream;
            video.play();
            
            // Capturar imagen
            setTimeout(function() {
                var canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                var contexto = canvas.getContext('2d');
                contexto.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convertir la imagen a base64
                var imagenBase64 = canvas.toDataURL('image/jpeg');

                // Enviar la imagen al servidor Flask (usando fetch API)
                fetch('/detectar-objetos', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ imagen: imagenBase64 })
                })
                .then(response => response.json())
                .then(data => {
                    // Manejar la respuesta del servidor (por ejemplo, mostrar los resultados de detección)
                    console.log(data);
                })
                .catch(error => console.error('Error:', error));
            }, 1000); // Esperar 1 segundo antes de capturar la imagen
        })
        .catch(function(error) {
            console.error('Error al acceder a la cámara:', error);
        });
                  }
