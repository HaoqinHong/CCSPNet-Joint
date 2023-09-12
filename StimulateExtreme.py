import xml.dom.minidom
import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

import albumentations as A

def prettyXml(element, indent, newline, level=0):
    """
    Formats XML data to make it more readable.
    """
    if element:
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)

    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level=level + 1)
    return element

def read_xml(ann_path):
    """
    Reads an XML file of an image and returns data in COCO dataset format.
    """
    in_file = open(ann_path, "r", encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()

    obj_dict = {
        "image": None,
        "bboxes": [],
        "class_labels": []
    }
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls == "waterweeds":
            continue

        xmlbox = obj.find('bndbox')

        obj_dict["bboxes"].append([
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymax').text)
        ])
        obj_dict["class_labels"].append(cls)

    return obj_dict

def save_xml(save_xml_path, image, bboxes, class_labels):
    """
    Saves data to an XML file in a specified format.
    """
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = "images"

    filename = ET.SubElement(root, "filename")
    filename.text = "filename"

    path = ET.SubElement(root, "path")
    path.text = "path"

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image.shape[1])
    height = ET.SubElement(size, "height")
    height.text = str(image.shape[0])
    depth = ET.SubElement(size, "depth")
    depth.text = str(image.shape[-1])

    segmented = ET.SubElement(root, "segmented")
    segmented.text = str(0)

    for i in range(len(bboxes)):
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")
        name.text = class_labels[i]
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = str(0)
        difficult = ET.SubElement(object, "difficult")
        difficult.text = str(0)

        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(bboxes[i][0])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(bboxes[i][1])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(bboxes[i][2])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(bboxes[i][3])

    root = prettyXml(root, '\t', '\n')
    tree = ET.ElementTree(root)
    tree.write(save_xml_path, encoding='utf-8')

def main():
    num = 0
    for img_name in tqdm(os.listdir(old_img_path)):
        if img_name.split(".")[0] not in train_list:
            continue

        img_path = os.path.join(old_img_path, img_name)
        if ".png" in img_name:
            xml_name = img_name.replace(".png", ".xml")
        elif ".jpg" in img_name:
            xml_name = img_name.replace(".jpg", ".xml")
        else:
            print("Unsupported file type:", img_name)
            exit()

        xml_path = os.path.join(old_xml_path, xml_name)

        ann = read_xml(xml_path)
        img = cv2.imread(img_path)
        ann["image"] = img
        
        print(xml_path)

        for i in range(len(transform_list)):
            aug_name = str(transform_list[i][0]).split("(")[0]
            ann_aug = transform_list[i](image=ann["image"], bboxes=ann["bboxes"], class_labels=ann["class_labels"])
            ann_aug["bboxes"] = [list(map(int, list(bbox))) for bbox in ann_aug["bboxes"]]

            save_img_name = "Aug_id_" + str(i) + "_" + aug_name + "_" + img_name
            save_xml_name = "Aug_id_" + str(i) + "_" + aug_name + "_" + xml_name 
            save_img_path = os.path.join(new_img_file, save_img_name)
            save_xml_path = os.path.join(new_aug_file, save_xml_name)

            cv2.imwrite(save_img_path, ann_aug["image"])
            save_xml(save_xml_path, ann_aug["image"], ann_aug["bboxes"], ann_aug["class_labels"])
            # print(save_img_path)

if __name__ == "__main__":
    # Albumentations documentation: https://albumentations.ai/docs/
    # How to use Albumentations: https://github.com/zk2ly/How-to-use-Albumentations
    classes = ["robot"]

    transform_list = [
        A.Compose(A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1), bbox_params=A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['class_labels'])),
        A.Compose(A.RandomFog(0.6, p=1), bbox_params=A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['class_labels'])),
        A.Compose(A.MotionBlur(p=1), bbox_params=A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['class_labels'])),
    ]

    old_xml_path = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC\VOC2007\Annotations"
    old_img_path = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC\VOC2007\JPEGImages"
    new_aug_file = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC_aug\VOC2007\Annotations"
    new_img_file = r"D:\ProgrammingProjects\PythonProjects\Context

