import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Definir el modelo de detección de lesiones deportivas
class SportsInjuryDetector(torch.nn.Module):
    def __init__(self, num_classes):
        super(SportsInjuryDetector, self).__init__()
        # Cargar el modelo pre-entrenado de Fast R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Reemplazar la capa de salida para ajustarse al número de clases
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("During training, targets should be passed")
        return self.model(images, targets)

# Función para crear el modelo
def create_model(num_classes):
    model = SportsInjuryDetector(num_classes)
    return model
