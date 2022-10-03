import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import to_torch,to_numpy,split_data_train_test
from Core.DataLoader import GoalCalibrationDataset
from Core.plottools import target_to_keypoints,batch_target_to_keypoints,plot_batch_keypoints
from utils import DATA_DIR

from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchsummary import summary
import sys
sys.path.append(r'C:\Users\Nikolaj\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\MasterThesis\Code\GoalCornerDetection\Core\torchhelpers')
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers import transforms, utils, engine, train
from Core.torchhelpers.utils import collate_fn
from Core.torchhelpers.engine import train_one_epoch, evaluate





if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Running on {device}')

    # initialize an instance of the dataloader
    # GoalData = GoalCalibrationDataset(DATA_DIR,transforms=get_transform(train=False))
    GoalData = GoalCalibrationDataset(DATA_DIR,transforms=None)

    # put dataloader into pytorch dataloader function. Batch size chosen to be 14 as it 854 is divisible by this
    train_loader,validation_loader = split_data_train_test(GoalData,validation_split=0.25,batch_size=2,shuffle_dataset=False,shuffle_seed=None,data_amount=0.1)




    # Setting hyper-parameters
    num_classes = 2 # 1 class (goal) + background

    from torchvision.models.detection.rpn import AnchorGenerator
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))

    # weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    # transforms = weights.transforms()
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.to(device)
    # print(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 2


    # gpu test
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, validation_loader, device)