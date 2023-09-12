import torch
import torch.nn as nn
from torch import nn, optim

import network
import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)


class Model_Joint(nn.Module):
    def __init__(self, weights, cfg, nc, hyp, resume):
        super(Model_Joint, self).__init__()

        # 4k model
        self.model_4k = network.B_transformer()
        model_4k_ckpt = torch.load(r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\4KDehazing-main\model\our_deblur40.pth")
        self.model_4k.load_state_dict(model_4k_ckpt)
        # model_4k.eval()
        # model_4k.to(device)

        # Model
        check_suffix(weights, '.pt')  # check weights
        weights = r"D:\ProgrammingProjects\PythonProjects\Contextual-Object-Detection\yolov5-7.0-pro\runs\train\exp_yolov5l-efficientvit-b2-cot_2\weights\best.pt"
        pretrained = weights.endswith('.pt')
        if pretrained:
            # with torch_distributed_zero_first(LOCAL_RANK):
            #     weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'))  # create
            exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))  # create
        self.model = model

    def forward(self, x, inference=False):
        # import cv2
        # import numpy as np
        # print(x)
        # save_x = x.mul(255).byte()
        # save_x = save_x.data.cpu().numpy().squeeze()
        # save_x = save_x.transpose(1,2,0)
        # save_x = cv2.cvtColor(save_x,cv2.COLOR_RGB2BGR)
        # save_x = save_x.astype(np.uint8)
        # print(save_x)
        # print(save_x.shape)
        # cv2.imwrite("test1.jpg", save_x)
        # exit()
        # output_4k = self.model_4k(x)
        # import cv2
        # import numpy as np
        # print(x)
        # save_x = output_4k.mul(255).byte()
        # save_x = save_x.data.cpu().numpy().squeeze()
        # save_x = save_x.transpose(1,2,0)
        # save_x = cv2.cvtColor(save_x,cv2.COLOR_RGB2BGR)
        # save_x = save_x.astype(np.uint8)
        # print(save_x)
        # print(save_x.shape)
        # cv2.imwrite("test2.jpg", save_x)
        # exit()
        # print("\n")
        # print("---------------")
        # print("inference:", inference)
        # print(output_4k.shape, x.shape)
        # print("\n")
        # exit()
        output = self.model(x, inference)
        return output, x