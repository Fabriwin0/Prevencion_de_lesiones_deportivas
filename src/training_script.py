import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Definir la función de entrenamiento
def train_model(dataset):
    # Cargar el modelo pre-entrenado de Fast R-CNN
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Configurar los parámetros de entrenamiento
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Entrenar el modelo
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in dataset:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}')
    
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'trained_model.pth')

# Llamar a la función de entrenamiento
if __name__ == "__main__":
    # Cargar el conjunto de datos de lesiones deportivas
    dataset = load_dataset('annotations/sports_injury_annotations.json', 'images/')
    
    # Entrenar el modelo
    train_model(dataset)
