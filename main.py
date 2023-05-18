from flask import Flask, request, jsonify
import torch
import io
import json
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_faster_rcnn_voc_model()

app = Flask(__name__)

model.load_state_dict(torch.load("/app/models/object_detection.pt"))
model.to(device)
model.eval()

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.route('/', methods=['GET'])
def welcome():
    return "Welcome to the Object Detection API!"

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    image = Image.open(request.files['file'].stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model(image_tensor)

    # Convert the detection output to a suitable JSON format
    formatted_detections = []
    for detection in detections:
        for label, box, score in zip(
            detection['labels'].tolist(),
            detection['boxes'].tolist(),
            detection['scores'].tolist()
        ):
            formatted_detections.append({
                'category_id': label,
                'bbox': box,
                'score': score,
            })

    return jsonify(formatted_detections)

if __name__ == '__main__':
    app.run()
