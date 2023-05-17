import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
import random

class ReducedVOC(Dataset):
    
    # Add the class_to_label dictionary as a class variable
    class_to_label = {'person': 1, 'bus': 2, 'car': 3, 'bird': 4, 'motorcycle': 5}

    def __init__(self, root_dir, transform=None, split="train", val_fraction=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.val_fraction = val_fraction

        self.annotations_dir = os.path.join(self.root_dir, 'reduced_Annotations')
        self.images_dir = os.path.join(self.root_dir, 'JPEGImages')

        self.image_files = sorted(os.listdir(self.images_dir))
        self.annotation_files = sorted(os.listdir(self.annotations_dir))

        # Shuffle and split the dataset
        combined = list(zip(self.image_files, self.annotation_files))
        random.seed(42)
        random.shuffle(combined)
        split_idx = int(len(combined) * (1 - self.val_fraction))

        if self.split == "train":
            combined = combined[:split_idx]
        elif self.split == "val":
            combined = combined[split_idx:]

        self.image_files, self.annotation_files = zip(*combined)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')

        # Load and parse annotation XML file
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            # Extract bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

            # Convert class name to integer label
            class_name = obj.find('name').text
            label = self.class_to_label[class_name]
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply image and bounding box transformations, if any
        if self.transform:
            img, boxes = self.transform(img, boxes)

        return img, boxes, labels

# Custom transform class for image and bounding box
class ResizeAndToTensor:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes):
        # Calculate the scaling factors for width and height
        width, height = img.size
        new_width, new_height = self.size
        width_scale = new_width / width
        height_scale = new_height / height

        # Resize the image
        img = F.resize(img, self.size)
        img = F.to_tensor(img)

        # Resize the bounding boxes
        boxes[:, 0] *= width_scale
        boxes[:, 1] *= height_scale
        boxes[:, 2] *= width_scale
        boxes[:, 3] *= height_scale

        return img, boxes


# Define train and validation transformations
train_transforms = transforms.Compose([
    ResizeAndToTensor(size=(300, 300)),
])

val_transforms = transforms.Compose([
    ResizeAndToTensor(size=(300, 300)),
])

# Create train and validation dataset instances with the defined transformations
train_dataset = ReducedVOC(root_dir='./data/VOCdevkit/VOC2007', transform=train_transforms, split="train")
val_dataset = ReducedVOC(root_dir='./data/VOCdevkit/VOC2007', transform=val_transforms, split="val")

# Create train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=None)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=None)


