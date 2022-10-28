import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Core/torchhelpers')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import to_torch,im_to_numpy,split_data_train_test
from Core.DataLoader import GoalCalibrationDatasetNEW
from Core.plottools import target_to_keypoints,batch_target_to_keypoints,plot_batch_keypoints,visualize
from utils import DATA_DIR

from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchsummary import summary
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers import transforms, utils, engine, train
from Core.torchhelpers.utils import collate_fn
from Core.torchhelpers.engine import train_one_epoch, evaluate
from torchvision.models.detection.rpn import AnchorGenerator

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Running on {device}')

    # initialize an instance of the dataloader
    GoalData = GoalCalibrationDatasetNEW(DATA_DIR,transforms=None)
    # put dataloader into pytorch dataloader function. Batch size chosen to be 14 as it 854 is divisible by this
    train_loader,validation_loader = split_data_train_test(GoalData,validation_split=0.25,batch_size=4,shuffle_dataset=True,shuffle_seed=None,data_amount=1)

    # Setting hyper-parameters
    num_classes = 2 # 1 class (goal) + background
    anchor_generator = AnchorGenerator(sizes=(64, 128, 256, 512, 1024), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.to(device)
    print(f'Model moved to device: {device}')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 1

    out_dict = {
        # total losses in each epoch
        'loss_keypoint_train': [],
        'loss_keypoint_val': [],
        'loss_all_train': [],
        'loss_all_val': [],
        # losses in each epoch
        'loss_keypoint_train_mean': [],
        'loss_keypoint_val_mean': [],
        'loss_all_train_mean': [],
        'loss_all_val_mean': []
    }
    # Run training loop
    for epoch in range(num_epochs):
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)

        # Add all losses for a given epoch to out_dict
        out_dict['loss_all_train'].append(metric_logger.meters['loss'].total)
        out_dict['loss_all_train_mean'].append(metric_logger.meters['loss'].global_avg)
        out_dict['loss_keypoint_train'].append(metric_logger.meters['loss_keypoint'].total)
        out_dict['loss_keypoint_train_mean'].append(metric_logger.meters['loss_keypoint'].global_avg)

        lr_scheduler.step()

####################### loop for validating
        # # Compute the validation accuracy
        # for images, targets in validation_loader:
        #     model.eval()
        #     with torch.no_grad():
        #         loss_dict = model(images, targets)
        #     losses = sum(loss for loss in loss_dict.values())
        #     # reduce losses over all GPUs for logging purposes
        #     loss_dict_reduced = utils.reduce_dict(loss_dict)
        #     losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # out_dict['loss_all_val'].append(losses_reduced)
        # out_dict['loss_all_val_mean'].append()

        # torch.cuda.empty_cache()
##################################

        coco_evaluator = evaluate(model, validation_loader, device)
        # coco_evaluator.coco_eval['bbox'].stats
        # coco_evaluator.coco_eval['keypoints'].stats


    print('we are done')    

    # torch.save(model.state_dict(), r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/model_states/first_train/kp_rcnn_weights_50epochs_newdata_bs4.pth')


if __name__ == '__main__':
    main()