import os 
import xml.etree.ElementTree as ET

input_folder = "./data/VOCdevkit/VOC2007/Annotations"
output_folder = "./data/VOCdevkit/VOC2007/reduced_Annotations"

selected_classes = {'person', 'car', 'bird', 'motorcycle'}

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(input_folder):
    tree = ET.parse(os.path.join(input_folder, file))
    root = tree.getroot()

    reduced_objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in selected_classes:
            reduced_objects.append(obj)

    if len(reduced_objects) > 0:
        for obj in root.findall('object'):
            if obj not in reduced_objects:
                root.remove(obj)
        tree.write(os.path.join(output_folder, file))

