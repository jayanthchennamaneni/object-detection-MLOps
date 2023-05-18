from flask import Flask, request, jsonify
import torch
import io
import json
from PIL import Image
from torchvision import transforms
from train import model

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("models/object_detection.pt"))
model.to(device)
model.eval()

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    image = Image.open(request.files['image'].stream).convert('RGB')
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