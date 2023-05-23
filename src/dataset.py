import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import Subset, DataLoader

# Dataset paths
data_root = './VOCdevkit/VOC2007'  # Root directory of the VOC dataset
train_set = 'train'  # Name of the train dataset
val_set = 'val'  # Name of the validation dataset

# Define a transformation to apply to images
data_transform = transforms.Compose([
    transforms.Resize((300, 300)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

# Define a mapping from class names to integers
class_name_to_label = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19,
}

# Define a collate function for data loading
def collate_fn(batch):
    images = []
    bboxes = []
    labels = []

    for sample in batch:
        images.append(sample[0])
        
        # Process annotations
        boxes = []
        class_labels = []
        objects = sample[1]['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
            
        for obj in objects:
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            
            label = class_name_to_label[obj['name']]
            class_labels.append(label)
        
        bboxes.append(torch.tensor(boxes, dtype=torch.float32))
        labels.append(torch.tensor(class_labels, dtype=torch.long))

    return torch.stack(images, 0), (bboxes, labels)


# Load train and validation datasets
train_dataset = VOCDetection(root=data_root, year='2007', image_set=train_set, transform=data_transform, download=False)
train_subset_size = 12  # Size of the train subset
train_subset = Subset(train_dataset, indices=range(train_subset_size))
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
val_dataset = VOCDetection(root=data_root, year='2007', image_set=val_set, transform=data_transform, download=False)
val_subset_size = 12  # Size of the validation subset
val_subset = Subset(val_dataset, indices=range(val_subset_size))
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)


if __name__ == "__main__":
    # Print the number of batches in the train_loader and val_loader
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")

    # Get the first batch from the train_loader
    images, annotations = next(iter(train_loader))

    # Convert images from tensors to numpy arrays
    images = images.numpy()

    # Plot the images
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        ax = fig.add_subplot(4, 1, i+1)
        image = images[i].transpose((1, 2, 0))  
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  
        image = np.clip(image, 0, 1)  
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
