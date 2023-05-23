import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.dataset import train_loader, class_name_to_label

# Parameters
num_epochs = 50 
learning_rate = 0.01  

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def get_faster_rcnn_voc_model(num_classes=21):
    """
    Get a Faster R-CNN model pre-trained on COCO dataset and replace the box predictor for VOC dataset.

    Args:
        num_classes (int): Number of classes in the VOC dataset.

    Returns:
        model: Faster R-CNN model with the box predictor replaced.
    """

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the box predictor to change the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

model = get_faster_rcnn_voc_model()  
model.to(device)  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

if __name__ == "__main__":
    # Training loop
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0
        for i, (images, (bboxes, labels)) in enumerate(train_loader):
            images = images.to(device)
            targets = []
            for j in range(len(bboxes)):
                target = {}
                target["boxes"] = bboxes[j].to(device)
                target["labels"] = labels[j].to(device)
                targets.append(target)

            optimizer.zero_grad()  
            loss_dict = model(images, targets)  
            losses = sum(loss for loss in loss_dict.values())  
            running_loss += losses.item()
            losses.backward()  
            optimizer.step()  

            if i % 1 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item()}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}")

    torch.save(model.state_dict(), "models/object_detection.pt")  