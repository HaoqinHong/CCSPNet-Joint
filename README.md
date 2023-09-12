This is the joint training model for traffic sign detection and image denoising proposed in our paper titled 
"CCSPNET-JOINT: Efficient Joint Training Method for Traffic Sign Detection under Extreme Conditions".


The image denoising module of our model utilizes the 4kDehazing model(cite: https://github.com/zzr-idam/4KDehazing.git), 
while the object detection module incorporates the improved model CCSPNet, 
which is based on the YOLOv5 baseline, as proposed in our article.

The repository includes:
1.CCSPNet model:
    CCSPNet-Joint/models/yolov5l-efficientvit-b2-cot.yaml

2.pretrained_pth:
    Download link：https://pan.baidu.com/s/1wfMUxK3Z09R00wus3XzVEA 
    Verification code：wor4
    Content:
    ccspnet-joint.pt
    our_deblur40.pth
    resnet50-0676ba61.pth

3.Dataset:
    CCTSDB: https://github.com/csust7zhangjm/CCTSDB.git
    Augument method for CCTSDB-AUG:  StimulateExtreme.py

3.CCSPNet-Joint/data/ours_aug.yaml

3.train_ccspnet_joint.py

4.detect_joint.py


