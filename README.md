This is the joint training model for traffic sign detection and image denoising proposed in our paper titled 
"CCSPNET-JOINT: Efficient Joint Training Method for Traffic Sign Detection under Extreme Conditions".


The image denoising module of our model utilizes the 4kDehazing model(cite: https://github.com/zzr-idam/4KDehazing.git), 
while the object detection module incorporates the improved model CCSPNet, 
which is based on the YOLOv5 baseline, as proposed in our article.

The proposed method and comparisons in this paper were conducted under a unified data augmentation approach. To replicate the experiments, you will need to download the dataset and pre-trained weights and place them in a specific directory. 
Then, in the terminal, run the command：python train_ccspnet_joint.py --rect

The repository includes:
1.CCSPNet model:
    CCSPNet-Joint/models/yolov5l-efficientvit-b2-cot.yaml

2.pretrained_pth:
    Download link：[https://pan.baidu.com/s/1wfMUxK3Z09R00wus3XzVEA](https://pan.baidu.com/s/1Vo-Xe07KtYYm5TF9Vx4DSQ) 
    Verification code：1rvo 
    Content:
    ccspnet-joint.pt
    our_deblur40.pth
    resnet50-0676ba61.pth

3.Dataset:
    CCTSDB: https://github.com/csust7zhangjm/CCTSDB.git
    Augment method for CCTSDB-AUG:  StimulateExtreme.py

4.CCSPNet-Joint/data/ours_aug.yaml

5.train_ccspnet_joint.py

6.detect_joint.py


