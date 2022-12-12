import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import numpy as np
from Core.DataLoader import train_transform
from Core.helpers import split_data_train_test, eval_PCK
from Core.wandbtools import make_PCK_plot_objects, prediction_outliers
from Core.plottools import plot_loss, get_prediction_images
# from Core.DataLoader import GoalCalibrationDataset, GoalCalibrationDataset4boxes
from utils import DATA_DIR, export_wandb_api
from torchvision.models.detection import keypointrcnn_resnet50_fpn
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers.utils import MetricLogger
from Core.torchhelpers.utils import reduce_dict
from Core.torchhelpers.engine import train_one_epoch, evaluate
from torchvision.models.detection.rpn import AnchorGenerator
import json
import wandb
import importlib
import time
from pathlib import Path

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

def save_model(save_folder, model, loss_dict, type):
    '''
    Save losses and weights of model
    type: String that is either 'best' or 'last'
    '''

    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    if type.lower() == 'last':
        weightsname = 'weights-last.pth'
        lossname = 'losses-last.json'
        message = f'Model weights and losses saved to {save_folder}'
    elif type.lower() == 'best':
        weightsname = 'weights-best.pth'
        lossname = 'losses-best.json'
        message = ''
    else:
        print(f'Expected argument type to be either "best" or "last", but {type} was given.')
        return
    # save losses
    with open(os.path.join(save_folder,lossname),'w') as file:
        json.dump(loss_dict, file, indent=4)
    torch.save(model.state_dict(), os.path.join(save_folder,weightsname))
    print(message)
    return

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Keypoint Detection Training", add_help=add_help)

    parser.add_argument("--data-dir", default=DATA_DIR, type=str, help="dataset directory path")
    parser.add_argument("-b", "--batch-size", default=4, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--validation-split", default=0.25, type=float, help="Fraction of the data to use as the validation set (float between 0 and 1)")
    parser.add_argument("--epochs", default=2, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 6)")
    parser.add_argument("--opt", default="adam", type=str, help="optimizer, either sgd or adam")
    parser.add_argument("--lr",default=0.001,type=float,help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum to use in the optimizer")
    parser.add_argument("--weight-decay", default=0.0005, type=float, dest="weight_decay", metavar="W", help="weight decay (default: 5e-4)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="/zhome/60/1/118435/Master_Thesis/Runs/", type=str, help="path to save outputs")
    parser.add_argument("--model-name", default="tester_model", type=str, help="Unique folder name for saving model results")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="If option is set, the script will only test the model")
    parser.add_argument("--load-path", type=str, help="path to load model checkpoint from. Must be present if --test-only is used")
    parser.add_argument("--data-aug", dest="data_aug", action="store_true", help="Augment data during training")
    parser.add_argument("--pckthreshup", default=200, type=int, help="Upper threshold on the pixel error when when calculating PCK")
    parser.add_argument("--predims_every", default=1, type=int, help="Interval (in epochs) in which to save intermittent prediction images from the validation set")
    parser.add_argument("--data-amount", default=1, type=float, help="fraction of data that should be used (float between 0 and 1)")
    parser.add_argument("--shuffle-dataset", choices=('True','False'), default='True', help="Shuffle which data ends up in train set and validation set. Default: True")
    parser.add_argument("--shuffle-epoch", choices=('True','False'), default='True', help="Shuffle data in each epoch. Default: True")
    parser.add_argument("--shuffle-dataset-seed", default=-1, type=int, help="Seed for shuffling dataset. Related to --shuffle-dataset. Default: -1, means no seed is set.")
    parser.add_argument("--shuffle-epoch-seed", default=-1, type=int, help="Seed for shuffling at every epoch. Related to --shuffle-epoch. Default: -1, means no seed is set.")
    parser.add_argument("--model-type", default="4box", choices=('1boxOLD','1box','4box'), help="Changes which type of model is run. Affects which dataloader is called and some parameters of the model. Default: 4box")
    parser.add_argument("--filter-data", choices=('True','False'), default='True', help="Shuffle which data ends up in train set and validation set. Default: True")

    return parser

class Params:
    def setparams(
        self,
        im_score_thresh,
        nms_iou_thresh,
        plot_kp_opaqueness,
        bbox_expand_x,
        bbox_expand_y):

        self.im_score_thresh = im_score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.plot_kp_opaqueness = plot_kp_opaqueness
        self.bbox_expand_x = bbox_expand_x
        self.bbox_expand_y = bbox_expand_y

    def setself(self, params):
        for name,val in params.items():
            self.__dict__[name] = val
    def __init__(self):
        self.setparams()

def main(args):
    # params = Params(im_score_thresh=0.7,nms_iou_thresh=0.3,plot_kp_opaqueness=0.4)
    # convert these arguments from strings to boolean
    args.shuffle_dataset = True if args.shuffle_dataset=='True' else False
    args.shuffle_epoch = True if args.shuffle_epoch=='True' else False
    args.filter_data = True if args.filter_data=='True' else False
    # Update the dataloader to import and use depending on the model_type chosen
    modeltype_to_dataloader = {
        "1boxOLD": "GoalCalibrationDatasetOLD",
        "1box":"GoalCalibrationDataset",
        "4box":"GoalCalibrationDataset4boxes"
        }
    module = importlib.__import__('Core.DataLoader', fromlist=[modeltype_to_dataloader[args.model_type]])
    DataClass = getattr(module, modeltype_to_dataloader[args.model_type])
    # set wandb api key as environment variable
    export_wandb_api()
    # initialize wandb run
    wandb.init(
        project="GoalCornerDetection",
        name=args.model_name,
        config={
            "original_model_name": args.model_name,
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.opt,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "data_amount": args.data_amount,
            "validation_split": args.validation_split,
            "shuffle_dataset": args.shuffle_dataset,
            "shuffle_dataset_seed": args.shuffle_dataset_seed,
            "shuffle_epoch": args.shuffle_epoch,
            "shuffle_epoch_seed": args.shuffle_epoch_seed,
            "filter_data": args.filter_data,
            "data_augmentation": args.data_aug,
            "workers": args.workers
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
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}')
    ### Set save path ###
    # save_folder = os.path.join(args.output_dir, args.model_name + f'_{args.epochs}epochs/')
    save_folder = str(Path(args.output_dir + '/'+args.model_name + f'_{args.epochs}epochs/'))
    # print options used in training
    print(f"""
    ####################
    Training parameters
    #################### 
    model_name={args.model_name}
    model_type={args.model_type}
    save_folder={save_folder}
    batch_size={args.batch_size}
    epochs={args.epochs}
    optimizer={args.opt}
    lr={args.lr}
    momentum={args.momentum}
    weight_decay={args.weight_decay}
    num_workers={args.workers}
    """)

    # initialize an instance of the dataloader class, one for train and one for validation
    if args.data_aug:
        GoalData_train = DataClass(args.data_dir, transforms=train_transform(), filter_data=args.filter_data)
    else:
        GoalData_train = DataClass(args.data_dir, transforms=None, filter_data=args.filter_data)

    GoalData_val = DataClass(args.data_dir, transforms=None, filter_data=args.filter_data)

    num_objects = GoalData_val.num_objectclasses_per_image # number of object classes in this model
    num_keypoints = GoalData_val.num_keypoints_per_object # number of keypoints to predict per object
    num_classes = GoalData_val.num_objectclasses_per_image + 1 # number of classes in keypoint r-cnn model. It is the number of object classes + 1 (background class)
    # put dataloader into pytorch dataloader function with batch loading
    train_loader,validation_loader = split_data_train_test(
                                                            GoalData_train,
                                                            GoalData_val,
                                                            validation_split=args.validation_split,
                                                            batch_size=args.batch_size,
                                                            data_amount=args.data_amount,
                                                            num_workers=args.workers,
                                                            shuffle_dataset=args.shuffle_dataset,
                                                            shuffle_dataset_seed=args.shuffle_dataset_seed,
                                                            shuffle_epoch = args.shuffle_epoch,
                                                            shuffle_epoch_seed=args.shuffle_epoch_seed,
                                                            pin_memory=False) # setting to True (makes it fail) speeds up host to device transfer when loading on cpu and pushing to gpu for training
    # Setting hyper-parameters
    # old anchor generator
    # anchor_generator = AnchorGenerator(sizes=(64, 128, 256, 512, 1024), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    # the different sizes to use for the anchor boxes
    anchor_sizes = ((64,), (128,), (256,), (384,), (512,))
    # list of possible aspect_ratios to use, due to different models being trained on different number of aspect_ratios
    aspect_ratios_all = (1.0, 4208/3120, 2.0, 2.5, 3.0, 0.5, 4.0)
    if args.test_only:
        # finds the number of aspect ratios used from the state_dict if loading a previously trained model
        state_dict = torch.load(args.load_path)
        number_aspect_ratios = len(state_dict.get('rpn.head.cls_logits.bias'))
        aspect_ratios_anchors = (aspect_ratios_all[:number_aspect_ratios], ) * len(anchor_sizes)
    else: # make an anchor generator with 3 aspect_ratios
        # which aspect ratios to use for every anchor size. Assumes aspect_ratio = height / width
        aspect_ratios_anchors = (aspect_ratios_all[:3], ) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios_anchors)
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=num_keypoints,rpn_anchor_generator=anchor_generator)
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
    
    if args.test_only:
        # torch.backends.cudnn.deterministic = True
        
        model.load_state_dict(state_dict)
        # evaluate(model, validation_loader, device=device)
        # Evaluate PCK for all the keypoints
        thresholds=np.arange(1,args.pckthreshup+1)
        PCK,pixelerrors = eval_PCK(model,validation_loader,device,thresholds=thresholds, num_objects=num_objects)
        # Log the PCK values in wandb
        PCK_plot_objects = make_PCK_plot_objects(PCK,thresholds)
        wandb.log(PCK_plot_objects)
        
        # Find the outliers in predictions and log them
        outliertable = prediction_outliers(pixelerrors, model, validation_loader, num_objects, device)
        wandb.log({"outliers_table": outliertable})
        print(f'Model has been tested!')
        return
    
    best_epoch = None
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

         # log epoch metrics to wandb dashboard
        metrics_epoch = {
            "epoch_metrics/loss_avg": {
                "train_all":metric_logger.meters['loss'].global_avg,
                "val_all":metric_logger_val.meters['loss'].global_avg,
                "train_keypoint":metric_logger.meters['loss_keypoint'].global_avg,
                "val_keypoint":metric_logger_val.meters['loss_keypoint'].global_avg
            },
            "epoch_metrics/loss_total": {
                "train_all":metric_logger.meters['loss'].total,
                "val_all":metric_logger_val.meters['loss'].total,
                "train_keypoint":metric_logger.meters['loss_keypoint'].total,
                "val_keypoint":metric_logger_val.meters['loss_keypoint'].total

            },
            "epoch": epoch}
        wandb.log(metrics_epoch)

        # save current best model (save if loss is lower than before)
        if epoch == 0:
            bestloss_epoch_val_avg = metric_logger_val.meters['loss'].global_avg
        if metric_logger_val.meters['loss'].global_avg < bestloss_epoch_val_avg:
            save_model(save_folder=save_folder,model=model,loss_dict=loss_dict, type='best')
            bestloss_epoch_val_avg = metric_logger_val.meters['loss'].global_avg
            print(f'New best validation loss achieved. Model saved at epoch {epoch}')
            best_epoch = epoch

        # get intermediate prediction images and log in wandb
        if epoch % args.predims_every == 0 or epoch == (args.epochs-1):
            # FIXME Consider which photos to show each time, if it should be the first case in the data_loader or what
            images,targets = next(iter(validation_loader))
            pred_images = get_prediction_images(model,images,targets,device, num_objects=num_objects,opaqueness=0.5)

            pred_images_dict = {f'Image_ID_{image_id}': wandb.Image(image_array, caption=f"Prediction at epoch {epoch}") for image_id,image_array in pred_images.items()}
            pred_images_dict['epoch'] = epoch
            wandb.log(pred_images_dict)

    print('\nFINISHED TRAINING :) #################################################################################\n')

    # get evaluation metrics, average precison and average recall for different IoUs or OKS thresholds
    # evaluate(model, validation_loader, device)
    ###################### save losses and weights from final epoch
    save_model(save_folder=save_folder, model=model, loss_dict=loss_dict, type='last')
    print(f'Best model achieved at epoch {best_epoch}.')

    # plot losses and save as images in save_folder
    # plot_loss(loss_dict,save_folder,args.epochs)
    
    # Evaluate PCK for all the keypoints
    thresholds=np.arange(1,args.pckthreshup+1)
    PCK,pixelerrors = eval_PCK(model,validation_loader,device,thresholds=thresholds, num_objects=num_objects)
    # Log the PCK values in wandb
    PCK_plot_objects = make_PCK_plot_objects(PCK,thresholds)
    wandb.log(PCK_plot_objects)
    
    # Find the outliers in predictions and log them
    outliertable = prediction_outliers(pixelerrors, model, validation_loader, num_objects, device)
    wandb.log({"outliers_table": outliertable})
def test(args):
    # just a test function
    print('before change')
    print(f'shuffle dataset: {args.shuffle_dataset}, type: {type(args.shuffle_dataset)}')
    print(f'shuffle dataset seed: {args.shuffle_dataset_seed}')
    print(f'shuffle epoch: {args.shuffle_epoch}, type: {type(args.shuffle_epoch)}')
    print(f'shuffle epoch seed: {args.shuffle_epoch_seed}')

    if args.shuffle_dataset:
        print('hej1')
    if args.shuffle_epoch:
        print('hej2')
    pass
    

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
    # test(args)