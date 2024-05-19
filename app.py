from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

app = Flask(__name__)

# Cargar el modelo preentrenado
def cargar_modelo():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Reemplazar el clasificador de la cabeza con un nuevo clasificador para las clases de lesiones deportivas
    num_classes = 81  # COCO dataset tiene 80 clases más una clase de fondo
    num_lesiones_deportivas = 5  # Por ejemplo, supongamos que hay 5 clases de lesiones deportivas
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_lesiones_deportivas)

    return model

# Función para procesar la imagen y ejecutar la detección de objetos
def detectar_lesiones_deportivas(imagen_base64, modelo):
    # Decodificar la imagen desde base64
    imagen_bytes = base64.b64decode(imagen_base64)
    imagen_np = np.frombuffer(imagen_bytes, dtype=np.uint8)
    imagen_cv2 = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)
    
    # Preprocesamiento de la imagen
    transformacion = transforms.Compose([transforms.ToTensor()])
    imagen_tensor = transformacion(imagen_cv2)
    
    # Ejecutar la detección de objetos
    modelo.eval()
    with torch.no_grad():
        resultados = modelo([imagen_tensor])
    
    # Filtrar detecciones de lesiones deportivas
    detecciones_lesiones_deportivas = []
    for i, deteccion in enumerate(resultados[0]['labels']):
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
    
    # Cargar el modelo
    modelo = cargar_modelo()
    
    # Detectar lesiones deportivas
    detecciones_lesiones = detectar_lesiones_deportivas(imagen_base64, modelo)
    
    return jsonify({'detecciones': detecciones_lesiones})

if __name__ == '__main__':
    app.run(debug=True)
