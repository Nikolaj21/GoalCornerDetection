import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import numpy as np
from Core.helpers import split_data_train_test, train_transform, eval_PCK
from Core.plottools import plot_loss
from Core.DataLoader import GoalCalibrationDataset,GoalCalibrationDatasetAUG
from utils import DATA_DIR, export_wandb_api
from torchvision.models.detection import keypointrcnn_resnet50_fpn
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers.utils import MetricLogger
from Core.torchhelpers.utils import reduce_dict
from Core.torchhelpers.engine import train_one_epoch, evaluate
from torchvision.models.detection.rpn import AnchorGenerator
import json
import wandb

# function to set api key as environment variable
export_wandb_api()

def validate_epoch(model, dataloader, device, epoch, print_freq):
    '''
    Run validation of all images and print losses
    '''
###################### loop for validating
    metric_logger_val = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    print(f'Running validation loop!')
    # Compute the validation loss
    batchnr = 0
    steps_per_epoch = len(dataloader)
    for images, targets in metric_logger_val.log_every(dataloader, print_freq, header):
        # move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            # Forward pass
            loss_dict = model(images, targets)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # log metrics to wandb dashboard
        metrics_val = {"validation/loss": losses_reduced,
                        "validation/loss_classifier":loss_dict_reduced['loss_classifier'],
                        "validation/loss_box_reg":loss_dict_reduced['loss_box_reg'],
                        "validation/loss_keypoint":loss_dict_reduced['loss_keypoint'],
                        "validation/loss_objectness":loss_dict_reduced['loss_objectness'],
                        "validation/loss_rpn_box_reg":loss_dict_reduced['loss_rpn_box_reg'],
                        "validation/step": steps_per_epoch*epoch+batchnr}
        wandb.log(metrics_val)
        batchnr += 1

        metric_logger_val.update(loss=losses_reduced, **loss_dict_reduced)
        # torch.cuda.empty_cache()
    return metric_logger_val

def save_model(save_folder, model, loss_dict):
    '''
    Save losses and weights of model
    '''
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    # save losses
    with open(os.path.join(save_folder,'losses.json'),'w') as file:
        json.dump(loss_dict, file, indent=4)
    torch.save(model.state_dict(), os.path.join(save_folder,'weights.pth'))
    print(f'Model weights and losses saved to {save_folder}')


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Keypoint Detection Training", add_help=add_help)

    parser.add_argument("--data-dir", default=DATA_DIR, type=str, help="dataset directory path")
    parser.add_argument("-b", "--batch-size", default=4, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--validation-split", default=0.25, type=float, help="How much of the data should be in the validation set (float between 0 and 1)")
    parser.add_argument("--epochs", default=2, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--workers", default=6, type=int, metavar="N", help="number of data loading workers (default: 6)")
    parser.add_argument("--opt", default="adam", type=str, help="optimizer, either sgd or adam")
    parser.add_argument("--lr",default=0.001,type=float,help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum to use in the optimizer")
    parser.add_argument("--weight-decay", default=0.0005, type=float, dest="weight_decay", metavar="W", help="weight decay (default: 5e-4)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/", type=str, help="path to save outputs")
    parser.add_argument("--model-name", default="tester_model", type=str, help="Unique folder name for saving model results")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    parser.add_argument("--data-aug", dest="data_aug", action="store_true", help="Augment data during training")

    return parser

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}')
    ### Set save path ###
    save_folder = args.output_dir + args.model_name + f'_{args.epochs}epochs/'
    # print options used in training
    print(f"""
    ####################
    Training parameters
    #################### 
    model_name={args.model_name}
    save_folder={save_folder}
    batch_size={args.batch_size}
    epochs={args.epochs}
    optimizer={args.opt}
    lr={args.lr}
    momentum={args.momentum}
    weight_decay={args.weight_decay}
    """)

    # initialize an instance of the dataloader class, one for train and one for validation
    if args.data_aug:
        GoalData_train = GoalCalibrationDatasetAUG(args.data_dir,transforms=train_transform(),istrain=True)
    else:
        GoalData_train = GoalCalibrationDatasetAUG(args.data_dir,transforms=None,istrain=False)

    GoalData_val = GoalCalibrationDatasetAUG(args.data_dir,transforms=None,istrain=False)

    # put dataloader into pytorch dataloader function with batch loading
    train_loader,validation_loader = split_data_train_test(
                                                            GoalData_train,
                                                            GoalData_val,
                                                            validation_split=args.validation_split,
                                                            batch_size=args.batch_size,
                                                            shuffle_dataset=True,
                                                            shuffle_seed=None,
                                                            data_amount=1,
                                                            num_workers=args.workers,
                                                            pin_memory=False) # pin_memory was false before running last training

    # Setting hyper-parameters
    num_classes = 2 # 1 class (goal) + background
    anchor_generator = AnchorGenerator(sizes=(64, 128, 256, 512, 1024), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.to(device)
    print(f'Model moved to device: {device}')
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = args.opt.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(params,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and Adam are supported.")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print_freq = args.print_freq

    loss_dict = {
        'train': {
            # losses in each step (batch)
            'keypoint': [],
            'all': [],
            # summary losses in each epoch
            'keypoint_mean': [],
            'all_mean': [],
            'keypoint_total': [],
            'all_total': []
        },
        'val': {
            # summary losses in each epoch
            'keypoint_mean': [],
            'all_mean': [],
            'keypoint_total': [],
            'all_total': []
        }
    }
    
    #FIXME remember to set it up so you load the model if you only want to evaluate a pretrained model
    if args.test_only:
        # torch.backends.cudnn.deterministic = True
        evaluate(model, validation_loader, device=device)
        return

    # initialize wandb run
    wandb.init(
        project="GoalCornerDetection",
        name=args.model_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.opt,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay
            }
        )
    # Define custom x-axis metric
    wandb.define_metric("train/step")
    # set all other train/ metrics to use this step
    wandb.define_metric("train/*", step_metric="train/step", summary="min")
    # Do the same, but for validation metrics
    wandb.define_metric("validation/step")
    wandb.define_metric("validation/*", step_metric="validation/step", summary="min")
    # Do the same, but for epoch metric
    wandb.define_metric("epoch")
    wandb.define_metric("epoch_metrics/loss_avg", step_metric="epoch", summary="min")
    wandb.define_metric("epoch_metrics/loss_total", step_metric="epoch", summary="min")
    

    ###################### Training ####################################
    for epoch in range(args.epochs):
        # Run training loop
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq)

        # Add all losses for a given epoch to loss_dict
        loss_dict['train']['all'].extend(list(metric_logger.meters['loss'].deque))
        loss_dict['train']['all_mean'].append(metric_logger.meters['loss'].global_avg)
        loss_dict['train']['all_total'].append(metric_logger.meters['loss'].total)
        loss_dict['train']['keypoint'].extend(list(metric_logger.meters['loss_keypoint'].deque))
        loss_dict['train']['keypoint_mean'].append(metric_logger.meters['loss_keypoint'].global_avg)
        loss_dict['train']['keypoint_total'].append(metric_logger.meters['loss_keypoint'].total)

        # take step in optimizer scheduler, updating current learning rate
        lr_scheduler.step()

        # Run testing loop
        metric_logger_val = validate_epoch(model, validation_loader, device, epoch, print_freq)

        # Add all losses for a given epoch to loss_dict
        loss_dict['val']['all_mean'].append(metric_logger_val.meters['loss'].global_avg)
        loss_dict['val']['all_total'].append(metric_logger_val.meters['loss'].total)
        loss_dict['val']['keypoint_mean'].append(metric_logger_val.meters['loss_keypoint'].global_avg)
        loss_dict['val']['keypoint_total'].append(metric_logger_val.meters['loss_keypoint'].total)

         # log metrics to wandb dashboard
        metrics_epoch = {
            "epoch_metrics/loss_avg": {
                "train":metric_logger.meters['loss_keypoint'].global_avg,
                "val":metric_logger_val.meters['loss_keypoint'].global_avg
            },
            "epoch_metrics/loss_total": {
                "train":metric_logger.meters['loss_keypoint'].total,
                "val":metric_logger_val.meters['loss_keypoint'].total
            },
            "epoch": epoch}
        wandb.log(metrics_epoch)

        if epoch % 5 == 0:
            pass

    print('\nFINISHED TRAINING :) #################################################################################\n')

    # get evaluation metrics, average precison and average recall for different IoUs or OKS thresholds
    evaluate(model, validation_loader, device)
    ###################### save losses and weights
    save_model(save_folder=save_folder, model=model, loss_dict=loss_dict)
    # plot losses and save as images in save_folder
    plot_loss(loss_dict,save_folder,args.epochs)
    # Evaluate PCK for all the keypoints
    thresholds=[10,30,50,75,100]
    PCK = eval_PCK(model,validation_loader,device,thresholds=thresholds)
    for pcktype,pck in PCK.items():
        print(f'Percentage of Correct Keypoints (PCK) {pcktype}\n{pck}')
    # Log the PCK values in wandb
    for threshold in thresholds:
        wandb.run.summary[f'PCK@{threshold}pix'] = PCK["all_corners"][threshold]
        wandb.run.summary[f'PCK_TL@{threshold}pix'] = PCK["top_left"][threshold]
        wandb.run.summary[f'PCK_TR@{threshold}pix'] = PCK["top_right"][threshold]
        wandb.run.summary[f'PCK_BL@{threshold}pix'] = PCK["bot_left"][threshold]
        wandb.run.summary[f'PCK_BR@{threshold}pix'] = PCK["bot_right"][threshold]
    

def test(args):
    # just a test function
    pass

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)