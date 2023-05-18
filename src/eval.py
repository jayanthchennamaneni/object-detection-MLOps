import torch
import torch.optim as optim
from train import model
from dataset import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

model.load_state_dict(torch.load("models/object_detection.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pascal_voc_to_coco(pascal_annotations):
    """
    Convert Pascal VOC annotations to COCO annotations format.

    Args:
        pascal_annotations (list): List of Pascal VOC annotations.

    Returns:
        dict: COCO annotations in dictionary format.
    """
    coco_annotations = {
        'categories': [],
        'images': [],
        'annotations': [],
    }

    # Define categories
    for label, class_name in enumerate(class_name_to_label.keys()):
        coco_annotations['categories'].append({'id': label, 'name': class_name})

    image_id = 0
    annotation_id = 0
    for image, targets in pascal_annotations:
        image_width, image_height = image.size
        image_info = {
            'file_name': str(image_id) + '.jpg',
            'id': image_id,
            'width': image_width,
            'height': image_height,
        }
        coco_annotations['images'].append(image_info)

        for target in targets:
            x_min, y_min, x_max, y_max = target['bbox']
            width = x_max - x_min
            height = y_max - y_min

            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': target['category_id'],
                'bbox': [x_min, y_min, width, height],
                'area': width * height,
                'iscrowd': 0,
            }
            coco_annotations['annotations'].append(annotation)
            annotation_id += 1

        image_id += 1

    return coco_annotations


def evaluate(model, val_loader):
    """
    Evaluate the model on the validation set using COCO evaluation metrics.

    Args:
        model: The trained object detection model.
        val_loader: DataLoader for the validation set.

    Returns:
        list: COCO evaluation statistics.
    """
    model.eval()
    all_annotations = []
    all_detections = []

    with torch.no_grad():
        for images, (bboxes, labels) in val_loader:
            images = images.to(device)
            for i in range(len(images)):
                image = images[i]
                targets = []
                for bbox, label in zip(bboxes[i], labels[i]):
                    target = {}
                    target['bbox'] = bbox.tolist()
                    target['category_id'] = label.tolist()
                    targets.append(target)
                all_annotations.append((image, targets))

            detections = model(images)
            for i, detection in enumerate(detections):
                for label, box, score in zip(
                    detection['labels'].tolist(),
                    detection['boxes'].tolist(),
                    detection['scores'].tolist()
                ):
                    all_detections.append({
                        'image_id': i,
                        'category_id': label,
                        'bbox': box,
                        'score': score,
                    })

    coco_gt = COCO()
    coco_gt.dataset = pascal_voc_to_coco(all_annotations)
    coco_gt.createIndex()

    if not all_detections:
        print("No detections were made.")
        return

    # Save and load detections in COCO format
    with open('detections.json', 'w') as f:
        json.dump(all_detections, f)
    coco_dt = coco_gt.loadRes('detections.json')

    # Evaluate the model using COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


if __name__ == "__main__":
    # Evaluate the model
    print("Evaluating on validation set...")
    coco_eval_stats = evaluate(model, val_loader)
    mAP = coco_eval_stats[0]
    AP_per_class = coco_eval_stats[1::5]

    print(f"Validation mAP: {mAP}")
    print(f"AP per class: {AP_per_class}")
