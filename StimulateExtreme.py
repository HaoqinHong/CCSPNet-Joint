import xml.dom.minidom
import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

import albumentations as A


def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行  
    if element:  # 判断element是否有子元素  
        if element.text == None or element.text.isspace(): # 如果element的text没有内容  
            element.text = newline + indent * (level + 1)    
        else:  
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)  
    #else:  # 此处两行如果把注释去掉，Element的text也会另起一行  
    #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level  
    temp = list(element) # 将elemnt转成list  
    for subelement in temp:  
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致  
            subelement.tail = newline + indent * (level + 1)  
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个  
            subelement.tail = newline + indent * level  
        prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作  
    return element

def read_xml(ann_path):
    """
    读取一张图片的xml返回coco数据集测试需要的格式。
    # [
    #     [class_1, x1, y1, x2, y2],
    #     [class_2, x1, y1, x2, y2],
    #     ......
    # ]
    {
        "image": image=None,
        "bboxes": [
            [x,y,x,y],
            [x,y,x,y]
        ]
        "category_id":[cls, cls]
    }
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
        # cls_id = classes.index(cls)
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
    tree.write(save_xml_path,  encoding = 'utf-8')


def main():
    num= 0
    for img_name in tqdm(os.listdir(old_img_path)):
        if img_name.split(".")[0] not in train_list:
            continue
        # 获取xml和对应的img的路径
        img_path = os.path.join(old_img_path, img_name)
        if ".png" in img_name:
            xml_name = img_name.replace(".png", ".xml")
        elif ".jpg" in img_name:
            xml_name = img_name.replace(".jpg", ".xml")
        else:
            print("类型不存在:", img_name)
            exit()
        xml_path = os.path.join(old_xml_path, xml_name)

        # 读取xml文件
        ann = read_xml(xml_path)
        # 读取img文件
        img = cv2.imread(img_path)
        ann["image"] = img
        
        print(xml_path)
        ## 转换并存储新xml和新img
        for i in range(len(transform_list)):
            aug_name = str(transform_list[i][0]).split("(")[0]
            ann_aug = transform_list[i](image=ann["image"], bboxes=ann["bboxes"], class_labels=ann["class_labels"])
            ann_aug["bboxes"] = [list(map(int, list(bbox))) for bbox in ann_aug["bboxes"]]

            save_img_name = "Aug_id_" + str(i) + "_" + aug_name + "_" + img_name
            save_xml_name = "Aug_id_" + str(i) + "_" + aug_name + "_" + xml_name 
            save_img_path = os.path.join(new_img_file, save_img_name)
            save_xml_path = os.path.join(new_aug_file, save_xml_name)

            # cv2.namedWindow("test", cv2.NORM_HAMMING)
            # cv2.imshow("test", ann_aug["image"])
            # cv2.namedWindow("test2", cv2.NORM_HAMMING)
            # cv2.imshow("test2", img)
            # cv2.waitKey(0)
            # exit()
            cv2.imwrite(save_img_path, ann_aug["image"])
            save_xml(save_xml_path, ann_aug["image"], ann_aug["bboxes"], ann_aug["class_labels"])
            # print(save_img_path)  

if __name__ == "__main__":
    # https://albumentations.ai/docs/
    # https://github.com/zk2ly/How-to-use-Albumentations
    classes = ["robot"]

    """ 
    (0)RandomShadow:模拟雨天
    (1)RandomFog:模拟雾天(0.6)
    (2)MotionBlur:模拟运动模糊
    """
    transform_list = [
        A.Compose(A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5,p=1),bbox_params=A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['class_labels'])),
        A.Compose(A.RandomFog(0.6,p=1),bbox_params=A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['class_labels'])),
        A.Compose(A.MotionBlur(p=1),bbox_params=A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['class_labels'])),
    ]

    old_xml_path = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC\VOC2007\Annotations" # 存放 xml 和 img 数据地址###需要修改的地方
    old_img_path = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC\VOC2007\JPEGImages"
    new_aug_file = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC_aug\VOC2007\Annotations" # 增强 xml 存放地址 ###需要修改的地方
    new_img_file = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC_aug\VOC2007\JPEGImages" #增强img存放地址 ###需要修改的地方

    train_path = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\CCTSDB_VOC\VOC2007\ImageSets\Main\test.txt"
    train_list = []
    with open(train_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            train_list.append(line.split("\n")[0])
    main()
