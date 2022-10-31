import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Core/torchhelpers')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import numpy as np
import matplotlib.pyplot as plt
from Core.helpers import split_data_train_test
from Core.DataLoader import GoalCalibrationDatasetNEW
from utils import DATA_DIR

from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchsummary import summary
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers import transforms, utils, engine, train
from Core.torchhelpers.utils import reduce_dict,SmoothedValue
from Core.torchhelpers.engine import train_one_epoch, evaluate
from torchvision.models.detection.rpn import AnchorGenerator

def validate_epoch(model, train_loader, device, epoch, print_freq):
###################### loop for validating
        metric_logger_val = utils.MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"

        print(f'Running testing loop!')
        # Compute the validation loss
        for images, targets in metric_logger_val.log_every(train_loader, print_freq, header):
        # for images, targets in validation_loader:
            # move images and targets to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                # Forward pass
                loss_dict = model(images, targets)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            metric_logger_val.update(loss=losses_reduced, **loss_dict_reduced)
            # torch.cuda.empty_cache()
        return metric_logger_val
#################################



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
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 5
    print_freq = 50
    model_name = f'kp_rcnn_v1_{num_epochs}epochs'

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
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq)

        # Add all losses for a given epoch to out_dict
        out_dict['loss_all_train'].append(metric_logger.meters['loss'].total)
        out_dict['loss_all_train_mean'].append(metric_logger.meters['loss'].global_avg)
        out_dict['loss_keypoint_train'].append(metric_logger.meters['loss_keypoint'].total)
        out_dict['loss_keypoint_train_mean'].append(metric_logger.meters['loss_keypoint'].global_avg)

        # take step in optimizer scheduler, updating current learning rate
        lr_scheduler.step()

        metric_logger_val = validate_epoch(model, validation_loader, device, epoch, print_freq)
        out_dict['loss_all_val'].append(metric_logger_val.meters['loss'].total)
        out_dict['loss_all_val_mean'].append(metric_logger_val.meters['loss'].global_avg)
        out_dict['loss_keypoint_val'].append(metric_logger_val.meters['loss_keypoint'].total)
        out_dict['loss_keypoint_val_mean'].append(metric_logger_val.meters['loss_keypoint'].global_avg)

    # get evaluation metrics, average precison and average recall for different IoUs
    evaluate(model, validation_loader, device)
    print('\nFINISHED TRAINING :)')    

    import json
    save_folder = f'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/{model_name}/'
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    
    # save losses
    with open(save_folder + f'losses.json','w') as file:
        json.dump(out_dict, file, indent=4)
    torch.save(model.state_dict(), save_folder + f'weights.pth')
    print(f'Model weights and losses saved to {save_folder}')


if __name__ == '__main__':
    main()