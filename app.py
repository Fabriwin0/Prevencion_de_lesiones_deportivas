from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import torch

app = Flask(__name__)

def cargar_modelo():
    ruta_modelo = 'src/model/fast_r-cnn_pytorch'
    model = torch.load(ruta_modelo)
    model.eval()
    return model

def detectar_lesiones_deportivas(imagen_base64, modelo):
    imagen_bytes = base64.b64decode(imagen_base64)
    imagen_np = np.frombuffer(imagen_bytes, dtype=np.uint8)
    imagen_cv2 = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)
    
    # Preprocesamiento de la imagen si es necesario
    
    with torch.no_grad():
        resultados = modelo.detect([imagen_cv2])
    
    detecciones_lesiones_deportivas = []
    for deteccion in resultados[0]['labels']:
        if deteccion.item() > 0:  # Ignorar la clase de fondo
            detecciones_lesiones_deportivas.append({
                'clase': deteccion.item(),
                'score': resultados[0]['scores'][i].item(),
                'bbox': resultados[0]['boxes'][i].tolist()
            })
    
    return detecciones_lesiones_deportivas

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detectar-lesiones', methods=['POST'])
def detectar_lesiones():
    data = request.get_json()
    imagen_base64 = data['imagen']
    
    modelo = cargar_modelo()
    
    detecciones_lesiones = detectar_lesiones_deportivas(imagen_base64, modelo)
    
    return jsonify({'detecciones': detecciones_lesiones})

if __name__ == '__main__':
    app.run(debug=True)
