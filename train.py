import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import numpy as np
from Core.DataLoader import train_transform
from Core.helpers import split_data_train_test, eval_PCK, PCK_auc, MSE_loss_corners
from Core.wandbtools import make_PCK_plot_objects, prediction_outliers
from Core.plottools import plot_loss, get_prediction_images
# from Core.DataLoader import GoalCalibrationDataset, GoalCalibrationDataset4boxes
from utils import DATA_DIR, export_wandb_api, DATA_DIR_TEST
from torchvision.models.detection import keypointrcnn_resnet50_fpn
# https://github.com/pytorch/vision/tree/main/references/detection
from Core.torchhelpers.utils import collate_fn
from Core.torchhelpers.engine import train_one_epoch, evaluate, validate_epoch
from torchvision.models.detection.rpn import AnchorGenerator
import json
import wandb
import importlib
import time
from pathlib import Path
from distutils.util import strtobool


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

class Params:
    def defaultparams(self):
        self.data_dir = DATA_DIR
        self.batch_size = 8
        self.validation_split = 0.25
        self.epochs = 5
        self.workers = 6
        self.opt = "adam"
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.print_freq = 100
        self.output_dir = "/zhome/60/1/118435/Master_Thesis/Scratch/s163848/Runs/"
        self.project_name = "GoalCornerDetection"
        self.model_name = None
        self.pckthreshup = 200
        self.predims_every = 5
        self.data_amount = 1
        self.shuffle_dataset = "True"
        self.shuffle_epoch = "False"
        self.shuffle_dataset_seed = 20
        self.shuffle_epoch_seed = -1
        self.model_type = "4box"
        self.filter_data = 'True'
        self.wandb_dir = '/zhome/60/1/118435/Master_Thesis/Scratch/s163848/'
        self.bbox_expand = 0.05
        self.bbox_expand_x = self.bbox_expand
        self.bbox_expand_y = self.bbox_expand

        self.test_only = False
        self.load_path = "No path"
        self.data_aug = False


    def setself(self, params):
        for name,val in params.items():
            self.__dict__[name] = val
    def __init__(self):
        self.defaultparams()

params = Params()

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Keypoint Detection Training", add_help=add_help)

    parser.add_argument("--data_dir", default=params.data_dir, type=str, help="dataset directory path")
    parser.add_argument("-b", "--batch_size", default=params.batch_size, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--validation_split", default=params.validation_split, type=float, help="Fraction of the data to use as the validation set (float between 0 and 1)")
    parser.add_argument("--epochs", default=params.epochs, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--workers", default=params.workers, type=int, metavar="N", help="number of data loading workers (default: 6)")
    parser.add_argument("--opt", default=params.opt, type=str, help="optimizer, either sgd or adam")
    parser.add_argument("--lr",default=params.lr,type=float,help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=params.momentum, type=float, metavar="M", help="momentum to use in the optimizer")
    parser.add_argument("--weight_decay", default=params.weight_decay, type=float, dest="weight_decay", metavar="W", help="weight decay (default: 5e-4)")
    parser.add_argument("--print_freq", default=params.print_freq, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=params.output_dir, type=str, help="path to save outputs")
    parser.add_argument("--project_name", default=params.project_name, type=str, help="Unique folder name for saving model results")
    parser.add_argument("--model_name", default=params.model_name, type=str, help="Unique folder name for saving model results")
    parser.add_argument("--pckthreshup", default=params.pckthreshup, type=int, help="Upper threshold on the pixel error when when calculating PCK")
    parser.add_argument("--predims_every", default=params.predims_every, type=int, help="Interval (in epochs) in which to save intermittent prediction images from the validation set")
    parser.add_argument("--data_amount", default=params.data_amount, type=float, help="fraction of data that should be used (float between 0 and 1)")
    parser.add_argument("--shuffle_dataset",  default=params.shuffle_dataset, choices=('True','False'), help="Shuffle which data ends up in train set and validation set. Default: True")
    parser.add_argument("--shuffle_epoch", default=params.shuffle_epoch, choices=('True','False'), help="Shuffle data in each epoch. Default: True")
    parser.add_argument("--shuffle_dataset_seed", default=params.shuffle_dataset_seed, type=int, help="Seed for shuffling dataset. Related to --shuffle-dataset. Default: -1, means no seed is set.")
    parser.add_argument("--shuffle_epoch_seed", default=params.shuffle_epoch_seed, type=int, help="Seed for shuffling at every epoch. Related to --shuffle-epoch. Default: -1, means no seed is set.")
    parser.add_argument("--model_type", default=params.model_type, choices=('1boxOLD','1box','4box'), help="Changes which type of model is run. Affects which dataloader is called and some parameters of the model. Default: 4box")
    parser.add_argument("--filter_data", default=params.filter_data, choices=('True','False'), help="Shuffle which data ends up in train set and validation set. Default: True")
    parser.add_argument("--wandb_dir", default=params.wandb_dir, type=str, help="Directory in which to make wandb folder for storing data from runs.")
    parser.add_argument("--bbox_expand_x", default=params.bbox_expand_x, type=float, help="relative width (x) of the gt bounding boxes in relation to the image dimensions. float [0,1]")
    parser.add_argument("--bbox_expand_y", default=params.bbox_expand_y, type=float, help="relative height (y) of the gt bounding boxes in relation to the image dimensions. float [0,1]")
    parser.add_argument("--bbox_expand", default=params.bbox_expand_y, type=float, help="relative height (y) and width (x) of the gt bounding boxes in relation to the image dimensions. float [0,1]")


    parser.add_argument("--test_only", dest="test_only", action="store_true", help="If option is set, the script will only test the model")
    parser.add_argument("--load_path", default=params.load_path, type=str, help="path to load model checkpoint from. Must be present if --test-only is used")
    parser.add_argument("--data_aug", dest="data_aug", action="store_true", help="Augment data during training")

    return parser



def main(config=params):
    # convert these arguments from strings to boolean
    config.shuffle_dataset = bool(strtobool(config.shuffle_dataset))
    config.shuffle_epoch = bool(strtobool(config.shuffle_epoch))
    config.filter_data = bool(strtobool(config.filter_data))
    # set wandb api key as environment variable
    export_wandb_api()
    # initialize wandb run
    run = wandb.init(project=config.project_name, name=config.model_name, config=config, dir=config.wandb_dir)
    args = wandb.config
    # change model_name to the wandb generated one, if relevant
    if not args.model_name:
        args.update({'model_name':run.name}, allow_val_change=True)

    def define_custom_metrics():
        # Define custom x-axis metric
        wandb.define_metric("train/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step", summary="min")
        # Do the same, but for validation metrics
        wandb.define_metric("validation/step")
        wandb.define_metric("validation/*", step_metric="validation/step", summary="min")
        # Do the same, but for epoch metric
        wandb.define_metric("epoch")
        wandb.define_metric("epoch_metrics/loss_avg_train_all", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_avg_val_all", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_avg_train_keypoint", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_avg_val_keypoint", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_total_train_all", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_total_val_all", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_total_train_keypoint", step_metric="epoch", summary="min")
        wandb.define_metric("epoch_metrics/loss_total_val_keypoint", step_metric="epoch", summary="min")
    define_custom_metrics()

    # Update the dataloader to import and use depending on the model_type chosen
    modeltype_to_dataloader = {
        "1boxOLD": "GoalCalibrationDatasetOLD",
        "1box":"GoalCalibrationDataset",
        "4box":"GoalCalibrationDataset4boxes"
        }
    module = importlib.__import__('Core.DataLoader', fromlist=[modeltype_to_dataloader[args.model_type]])
    DataClass = getattr(module, modeltype_to_dataloader[args.model_type])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}')
    ### Set save path ###
    save_folder = str(Path(args.output_dir + run.name + f'_{args.epochs}epochs/'))
    # print options used in training
    print(f"""
    ####################
    Training parameters
    #################### 
    model_name={run.name}
    model_type={args.model_type}
    save_folder={save_folder}
    batch_size={args.batch_size}
    epochs={args.epochs}
    optimizer={args.opt}
    lr={args.lr}
    momentum={args.momentum}
    weight_decay={args.weight_decay}
    num_workers={args.workers}
    filter_data={args.filter_data}
    data_aug={args.data_aug}
    """)

    # initialize an instance of the dataloader class, one for train and one for validation
    if args.data_aug:
        GoalData_train = DataClass(args.data_dir, transforms=train_transform(), filter_data=args.filter_data, config=args)
    else:
        GoalData_train = DataClass(args.data_dir, transforms=None, filter_data=args.filter_data, config=args)

    GoalData_val = DataClass(args.data_dir, transforms=None, filter_data=args.filter_data, config=args)

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
    if args.model_type == '1box':
        anchor_sizes = ((860,), (1608,), (2110,), (2448,) ,(2836,))
        num_aspect_ratios = 5
    # the different sizes to use for the anchor boxes
    elif args.model_type == '4box':
        anchor_sizes = ((64,), (128,), (256,), (384,), (512,))
        num_aspect_ratios = 1
    print(f'anchor_sizes: {anchor_sizes}')
    print(f'image shape: {GoalData_train[0][0].shape}')
    # list of possible aspect_ratios to use, due to different models being trained on different number of aspect_ratios
    aspect_ratios_all = (4208/3120, 1.0, 2.0, 2.5, 3.0, 0.5, 4.0)
    # aspect_ratios_all = (1.2, 1.7, 2.4, 2.6, 3.0) # test aspect_ratios
    if args.test_only:
        torch.backends.cudnn.deterministic = True
        # finds the number of aspect ratios used from the state_dict if loading a previously trained model
        state_dict = torch.load(args.load_path)
        number_aspect_ratios = len(state_dict.get('rpn.head.cls_logits.bias'))
        aspect_ratios_anchors = (aspect_ratios_all[:number_aspect_ratios], ) * len(anchor_sizes)
        
        # make dataloader for test set (when testing, should input test dataset as args.data_dir if you want to test on test set, not validation set)
        # GoalData_test = DataClass(DATA_DIR_TEST, transforms=None, filter_data=args.filter_data, config=args)
        # my functions expect a dataloader defined from a torch.utils.data.Subset
        indices = list(range(len(GoalData_val)))
        GoalData_subset = torch.utils.data.Subset(GoalData_val,indices)
        test_loader = torch.utils.data.DataLoader(GoalData_subset,
                                                   batch_size=args.batch_size,
                                                   collate_fn=collate_fn, 
                                                   num_workers=args.workers, 
                                                   pin_memory=False, 
                                                   shuffle=args.shuffle_epoch)
        # load model
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios_anchors)
        model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=num_keypoints,rpn_anchor_generator=anchor_generator)
        model.to(device)
        model.load_state_dict(state_dict)
        # Evaluate PCK for all the keypoints
        thresholds=np.arange(1,args.pckthreshup+1)
        PCK,pixelerrors = eval_PCK(model,test_loader,device,thresholds=thresholds, num_objects=num_objects)
        # Log the PCK values in wandb
        PCK_plot_objects = make_PCK_plot_objects(PCK,thresholds)
        wandb.log(PCK_plot_objects)
        
        # Find the outliers in predictions and log them
        outliertable = prediction_outliers(pixelerrors, model, test_loader, num_objects, device)
        wandb.log({"outliers_table": outliertable})
        print(f'Model has been tested!')

        # get evaluation metrics, average precison and average recall for different IoUs or OKS thresholds
        # evaluate(model, validation_loader, device)
        return

    else: # make an anchor generator with 3 aspect_ratios (1 for now) (for 1box: all for now)
        # which aspect ratios to use for every anchor size. Assumes aspect_ratio = height / width
        aspect_ratios_anchors = (aspect_ratios_all[:num_aspect_ratios], ) * len(anchor_sizes)
    print(f'aspect_ratios: {aspect_ratios_anchors}')
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
    
    best_epoch = None
    ###################### Training ####################################
    for epoch in range(args.epochs):
        # Run training loop
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, args.print_freq)

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
        metric_logger_val = validate_epoch(model, validation_loader, device, epoch, args.print_freq)

        loss_epoch_val_keypoint = metric_logger_val.meters['loss_keypoint'].global_avg

        # Add all losses for a given epoch to loss_dict
        loss_dict['val']['all_mean'].append(metric_logger_val.meters['loss'].global_avg)
        loss_dict['val']['all_total'].append(metric_logger_val.meters['loss'].total)
        loss_dict['val']['keypoint_mean'].append(loss_epoch_val_keypoint)
        loss_dict['val']['keypoint_total'].append(metric_logger_val.meters['loss_keypoint'].total)

        metrics_epoch = {
            "epoch_metrics/loss_avg_train_all": metric_logger.meters['loss'].global_avg,
            "epoch_metrics/loss_avg_val_all": metric_logger_val.meters['loss'].global_avg,
            "epoch_metrics/loss_avg_train_keypoint": metric_logger.meters['loss_keypoint'].global_avg,
            "epoch_metrics/loss_avg_val_keypoint": loss_epoch_val_keypoint,
        
            "epoch_metrics/loss_total_train_all": metric_logger.meters['loss'].total,
            "epoch_metrics/loss_total_val_all": metric_logger_val.meters['loss'].total,
            "epoch_metrics/loss_total_train_keypoint": metric_logger.meters['loss_keypoint'].total,
            "epoch_metrics/loss_total_val_keypoint": metric_logger_val.meters['loss_keypoint'].total,
            "epoch": epoch
            }
        wandb.log(metrics_epoch)

        # save current best model (save if loss is lower than before)
        if epoch == 0:
            best_epoch = 0
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

    # calculate AUC of PCK plots and log
    pck_auc_dict = PCK_auc(PCK, thresholds)
    pck_auc_logs = {f'pck_auc_{cat}': auc for cat,auc in pck_auc_dict.items()}
    wandb.log(pck_auc_logs)
    # caclulate MSE of predictions and log
    mse_dict = MSE_loss_corners(pixelerrors)
    mse_logs = {f'MSE_loss_{cat}': mse for cat,mse in mse_dict.items()}
    wandb.log(mse_logs)

    # Find the outliers in predictions and log them
    outliertable = prediction_outliers(pixelerrors, model, validation_loader, num_objects, device)
    wandb.log({"outliers_table": outliertable})
    # log mean error for sweep purpose
    wandb.log({"mean_error_all": outliertable.get_column("mean",convert_to="numpy")[0]})

    wandb.finish()
    return

def test(args=None):
    # This is just a test function
    if not args:
        print('\n#########################args is None\n#########################')
    import yaml
    import random
    def train_one_epoch(epoch, lr, bs): 
        acc = 0.25 + ((epoch/30) +  (random.random()/10))
        loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
        return acc, loss

    def evaluate_one_epoch(epoch): 
        acc = 0.1 + ((epoch/20) +  (random.random()/10))
        loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
        return acc, loss

    # with open('./testsweep.yaml','r') as file:
    #     config = yaml.load(file,Loader=yaml.FullLoader)
    run = wandb.init()

    # note that we define values from `wandb.config` instead 
    # of defining hard values
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

    wandb.log({
        'epoch': epoch, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
    })

    return 

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)