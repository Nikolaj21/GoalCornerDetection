
import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f'Running on {device}')
from Core.DataLoader import GoalCalibrationDataset,GoalCalibrationDatasetOLD
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from Core.helpers import split_data_train_test
from Core.plottools import get_prediction_images
import wandb
from utils import DATA_DIR, export_wandb_api
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

def make_PCK_plot_objects(PCK,thresholds):
    PCK_plot_objects = {}
    for pck_type,pck_values in PCK.items():
        data = [[x,y] for x,y in zip(thresholds,pck_values.values())]
        table = wandb.Table(data=data, columns = ["Threshold", "PCK"])
        # wandb.log({f"PCK_{pck_type}" : wandb.plot.line(table, "Threshold", "PCK", title=f"PCK_{pck_type} Curve")})
        PCK_plot_objects[f"PCK_{pck_type} Curve"] = wandb.plot.line(table, "Threshold", "PCK", title=f"PCK_{pck_type} Curve")
    return PCK_plot_objects

def prediction_outliers(errors_dict, model, data_loader, num_objects, device):
    '''
    Finds some summary statistics in regards to the errors of the predictions. Also save the prediction that are classified as outliers and their predicted images for logging
    args:
        data_loader: assumes a pytorch dataloader that is defined with a torch.util.data.Subset 
    '''
    data = []
    for cat,metrics in errors_dict.items():
        if len(metrics) > 0:
            # extracts the errors list and discards the image ids and labels
            _,_,errors = zip(*metrics)
        else:
            print(f'category: {cat} did not have any elements, skipping...')
            continue
        # find the missing predictions, i.e. when a corner is not predicted at all in an image
        missing_tuplist = [(image_id,label,error) for image_id,label,error in metrics if error is None]
        num_missing_preds = len(missing_tuplist)
        if num_missing_preds > 0:
            missing_ids, missing_labels, _ = zip(*missing_tuplist)
            # filter duplicate missing_ids, but keep all missing_labels that way we don't get duplicate missing_predims
            missing_ids = tuple(set(missing_ids))
            data_missing = [data_loader.dataset.dataset.__getitem__(i) for i in missing_ids]
            missingimages,missingtargets = zip(*data_missing)
            missing_predims = get_prediction_images(model=model,images=missingimages,targets=missingtargets,device=device,num_objects=num_objects)
            missing_predims_list = [wandb.Image(image_array, caption=f"Prediction with missing corner(s), Image ID: {image_id}") for image_id,image_array in missing_predims.items()]
        else:
            missing_ids, missing_predims_list, missing_labels = (),(),()
        Ndata = len(errors) - num_missing_preds
        minval = np.min(errors)
        maxval = np.max(errors)
        std = np.std(errors)
        mean = np.mean(errors)
        median = np.median(errors)
        # make sure the inliner_min doesn't go below 0
        inlier_min = np.maximum(mean-3*std,0)
        inlier_max = mean+3*std
        outliers_tuplist = [(image_id,label,error) for image_id,label,error in metrics if not inlier_min <= error <= inlier_max]
        num_outliers = len(outliers_tuplist)
        pct_outliers = num_outliers / Ndata
        if num_outliers > 0:
            outlier_ids, outlier_labels, outliers = zip(*outliers_tuplist)
            # filter duplicate outlier_ids, but keep all outlier_labels and outliers (error values) that way we don't get duplicate outlier_predims
            outlier_ids = tuple(set(outlier_ids))
            data_plot = [data_loader.dataset.dataset.__getitem__(i) for i in outlier_ids]
            outlierimages,outliertargets = zip(*data_plot)
            outlier_predims = get_prediction_images(model=model,images=outlierimages,targets=outliertargets,device=device,num_objects=num_objects)
            outlier_predims_list = [wandb.Image(image_array, caption=f"Prediction Outlier, Image ID: {image_id}") for image_id,image_array in outlier_predims.items()]
        else:
            outliers, outlier_ids, outlier_predims_list, outlier_labels = (),(),(),()
        data.append((cat, Ndata, minval, maxval, std, mean, median, inlier_min, inlier_max, num_outliers, pct_outliers, outliers, outlier_ids, outlier_predims_list, outlier_labels, num_missing_preds, missing_ids, missing_predims_list, missing_labels))
    table = wandb.Table(data=data, columns =['cat', 'Ndata', 'min', 'max', 'std', 'mean', 'median', 'inlier_min', 'inlier_max', '#outliers', '%outliers', 'outliers', 'outlier_ids', 'outlier_ims', 'outlier_labels', '#missing_corners', 'missing_ids', 'missing_ims', 'missing_labels'])
    return table

def wandbapi_load_run(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    return run

def log_wandb_summary(objects,run):

    # Log the objects in wandb
    print('###############################')
    for obj_name, obj in objects.items():
        print(f' logging object: {obj_name}')
        run.summary[obj_name] = obj

    run.summary.update()
    print('\n All objects logged! :))')
    return

def make_PCK_plots(PCK,thresholds):
    for pck_type,pck_values in PCK.items():
        pck_values = list(pck_values.values())
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(thresholds,pck_values)
        ax.set_title(f'PCK curve for type: {pck_type}')
        ax.set_xlabel('threshold')
        ax.set_ylabel('PCK')
        fig.savefig(f'PCK_{pck_type}.png')
    return

def main():
    # remember to update with correct run path, which can be found under overview section of a run
    run_path = "nikolaj21/GoalCornerDetection/1k9srkii" # path for tester_sgd_da_50epochs
    run = wandbapi_load_run(run_path)
    _,validation_loader = load_data(DATA_DIR)
    # remember to update with correct model weights.pth
    load_path = r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/tester_sgd_da_50epochs/weights.pth'
    model = load_model(load_path)
    # # set thresholds for PCK evaluation
    # thresholds=np.arange(1,101)
    # PCK = run_PCK(model,validation_loader,thresholds)
    # # make_PCK_plots(PCK,thresholds)

    # wandb_objects = make_PCK_plot_objects(PCK,thresholds)

    # log_wandb_summary(wandb_objects,run)


def load_model(load_path):
    anchor_generator = AnchorGenerator(sizes=(128, 256, 512, 1024, 2048), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=2, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.load_state_dict(torch.load(load_path))
    model.to(device)
    model.eval()
    print(f'Model loaded!')
    return model

def load_data(data_path):
    # initialize an instance of the dataloader
    GoalData = GoalCalibrationDataset(data_path,transforms=None,istrain=False)
    GoalData_val = GoalCalibrationDataset(data_path,transforms=None,istrain=False)
    # put dataloader into pytorch dataloader
    train_loader,validation_loader = split_data_train_test(GoalData,GoalData_val,validation_split=0.25,batch_size=4,shuffle_dataset=True,shuffle_seed=None,data_amount=1)

    print('Data loaded!')
    return train_loader,validation_loader

def make_UA_PCK_curve():
    # export wandb api key to environment variable
    export_wandb_api()
    
    shuffle_dataset = True
    shuffle_dataset_seed = 10
    shuffle_epoch = False
    shuffle_epoch_seed = -1
    data_amount = 1
    validation_split=0.25
    wandb.init(
        project="GoalCornerDetection",
        name="UserAnnotations",
        config={
            "data_amount": data_amount,
            "validation_split": validation_split,
            "shuffle_dataset": shuffle_dataset,
            "shuffle_dataset_seed": shuffle_dataset_seed,
            "shuffle_epoch": shuffle_epoch,
            "shuffle_epoch_seed": shuffle_epoch_seed
            }
        )
    # initialize an instance of the dataloader
    GoalDataGT = GoalCalibrationDataset(DATA_DIR,transforms=None)
    GoalDataUA = GoalCalibrationDatasetOLD(DATA_DIR,transforms=None)
    # put dataloader into pytorch dataloader
    _,GT_loader = split_data_train_test(
                                        GoalDataGT,
                                        GoalDataGT,
                                        validation_split=validation_split,
                                        batch_size=8,
                                        data_amount=data_amount,
                                        num_workers=6,
                                        shuffle_dataset=shuffle_dataset,
                                        shuffle_dataset_seed=shuffle_dataset_seed,
                                        shuffle_epoch=shuffle_epoch,
                                        shuffle_epoch_seed=shuffle_epoch_seed)
    _,UA_loader = split_data_train_test(
                                        GoalDataUA,
                                        GoalDataUA,
                                        validation_split=validation_split,
                                        batch_size=8,
                                        data_amount=data_amount,
                                        num_workers=6,
                                        shuffle_dataset=shuffle_dataset,
                                        shuffle_dataset_seed=shuffle_dataset_seed,
                                        shuffle_epoch=shuffle_epoch,
                                        shuffle_epoch_seed=shuffle_epoch_seed)
    num_objects = 1
    pixelerrors_all = []
    print(f'Finding pixelerror for all predictions...')
    start_time = time.time()
    # Run through all images and get the pixel distance (error) between predictions and ground-truth
    for (_,targetsGT),(_,targetsUA) in zip(GT_loader, UA_loader):
        # move outputs to cpu
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # extract the euclidean distance error (in pixels) between every ground-truth and detection keypoint in the batch. Also return image_ids and labels for every distance measure for reference
        for targetGT, targetUA in zip(targetsGT, targetsUA):
            label_to_gts = {}
            label_to_dts = {}
            # make a dictionary for the gts and dts in every target and output that save the label, keypoints and scores, for later sorting
            for labelGT,kpGT,labelUA,kpUA in zip(targetGT['labels'],targetGT['keypoints'],targetUA['labels'],targetUA['keypoints']):
                label_to_gts[labelGT.item()] = kpGT
                label_to_dts[labelUA.item()] = kpUA
            # compare the gt and dt of every object with the same label, taking only the highest scored one
            for label in range(1,num_objects+1):
                # get the obj_gt and obj_dt for this label (obj_dt may not exist)
                obj_gt = label_to_gts[label]
                obj_dt = label_to_dts[label]
                # find the distance between every gt and gt for this label, and add to list of distances, along witht the image_id
                for gt,dt in zip(obj_gt,obj_dt):
                    pixelerrors_all.append((targetUA['image_id'], label, np.linalg.norm(dt[:2]-gt[:2])))
    num_keypoints = 4
    pixelerrors_TL = pixelerrors_all[0::num_keypoints]
    pixelerrors_TR = pixelerrors_all[1::num_keypoints]
    pixelerrors_BL = pixelerrors_all[2::num_keypoints]
    pixelerrors_BR = pixelerrors_all[3::num_keypoints]
    pixelerrors = {
        "all":pixelerrors_all,
        "TL":pixelerrors_TL,
        "TR":pixelerrors_TR,
        "BL": pixelerrors_BL,
        "BR": pixelerrors_BR
        }
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total time: {total_time_str}')

    # count the number of correctly classified keypoints according to every threshold
    print(f'Running PCK evaluation on all thresholds...')
    start_time = time.time()
    N_ims = len(UA_loader.dataset.indices)
    total_keypoints = {'all':N_ims*4, 'TL':N_ims, 'TR':N_ims, 'BL':N_ims, 'BR':N_ims}
    thresholds = np.arange(1,201)
    PCK = {
        key:{threshold: np.count_nonzero([error < threshold for _,_,error in errors]) / total_keypoints[key] for threshold in thresholds}
        for key,errors in pixelerrors.items()
        }

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total time: {total_time_str} ({total_time/len(thresholds):.4f} s / threshold)')

    PCK_plot_objects = make_PCK_plot_objects(PCK,thresholds)
    wandb.log(PCK_plot_objects)


if __name__ == '__main__':
    main()
    # make_UA_PCK_curve()