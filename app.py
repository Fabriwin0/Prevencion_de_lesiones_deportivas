# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms

app = Flask(__name__)

# Cargar el modelo preentrenado
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()

# Clase de COCO dataset
classes = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
]

# Función para dibujar bounding boxes
def draw_bboxes(image: np.array, det_objects: dict):
    """Draw bounding boxes and predicted classes"""
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for box, box_class, score in zip(det_objects[0]['boxes'].detach().numpy().astype(int),
                                     det_objects[0]['labels'].detach().numpy(),
                                     det_objects[0]['scores'].detach().numpy()):
        if score > 0.5:
            box = [(box[0], box[1]), (box[2], box[3])]
            cv2.rectangle(img=image, pt1=box[0], pt2=box[1], color=colors[box_class], thickness=4)
            cv2.putText(img=image, text=classes[box_class], org=box[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=colors[box_class], thickness=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detectar-objetos', methods=['POST'])
def detectar_objetos():
    data = request.get_json()
    image_data = data['imagen'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    transform = transforms.ToTensor()
    nn_input = transform(img)
    detected_objects = model([nn_input])

    draw_bboxes(img, detected_objects)

    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({'imagen': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
