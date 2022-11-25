
import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f'Running on {device}')
from Core.DataLoader import GoalCalibrationDataset,GoalCalibrationDatasetOLD
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from Core.helpers import eval_PCK,split_data_train_test
import wandb
from utils import DATA_DIR, export_wandb_api
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime


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
    GoalData = GoalCalibrationDatasetAUG(data_path,transforms=None,istrain=False)
    GoalData_val = GoalCalibrationDatasetAUG(data_path,transforms=None,istrain=False)
    # put dataloader into pytorch dataloader
    train_loader,validation_loader = split_data_train_test(GoalData,GoalData_val,validation_split=0.25,batch_size=4,shuffle_dataset=True,shuffle_seed=None,data_amount=1)

    print('Data loaded!')
    return train_loader,validation_loader

def run_PCK(model,data_loader,thresholds): 
    print('\nRunning PCK eval...')
    PCK,_ = eval_PCK(model,data_loader,device,thresholds=thresholds)
    # for pcktype,pck in PCK.items():
    #     print(f'Percentage of Correct Keypoints (PCK) {pcktype}\n{pck}')
    return PCK

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

def make_PCK_plot_objects(PCK,thresholds):
    PCK_plot_objects = {}
    for pck_type,pck_values in PCK.items():
        data = [[x,y] for x,y in zip(thresholds,pck_values.values())]
        table = wandb.Table(data=data, columns = ["Threshold", "PCK"])
        # wandb.log({f"PCK_{pck_type}" : wandb.plot.line(table, "Threshold", "PCK", title=f"PCK_{pck_type} Curve")})
        PCK_plot_objects[f"PCK_{pck_type} Curve"] = wandb.plot.line(table, "Threshold", "PCK", title=f"PCK_{pck_type} Curve")
    return PCK_plot_objects

def main():
    # remember to update with correct run path, which can be found under overview section of a run
    run_path = "nikolaj21/GoalCornerDetection/1k9srkii" # path for tester_sgd_da_50epochs
    run = wandbapi_load_run(run_path)
    _,validation_loader = load_data(DATA_DIR)
    # remember to update with correct model weights.pth
    load_path = r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/tester_sgd_da_50epochs/weights.pth'
    model = load_model(load_path)
    # set thresholds for PCK evaluation
    thresholds=np.arange(1,101)
    PCK = run_PCK(model,validation_loader,thresholds)
    # make_PCK_plots(PCK,thresholds)

    wandb_objects = make_PCK_plot_objects(PCK,thresholds)

    log_wandb_summary(wandb_objects,run)


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
    GoalDataGT = GoalCalibrationDataset(DATA_DIR,transforms=None,istrain=False)
    GoalDataUA = GoalCalibrationDatasetOLD(DATA_DIR,transforms=None,istrain=False)
    # put dataloader into pytorch dataloader
    _,GT_loader = split_data_train_test(
                                        GoalDataGT,
                                        GoalDataGT,
                                        validation_split=validation_split,
                                        batch_size=4,
                                        data_amount=data_amount,
                                        num_workers=3,
                                        shuffle_dataset=shuffle_dataset,
                                        shuffle_dataset_seed=shuffle_dataset_seed,
                                        shuffle_epoch=shuffle_epoch,
                                        shuffle_epoch_seed=shuffle_epoch_seed)
    _,UA_loader = split_data_train_test(
                                        GoalDataUA,
                                        GoalDataUA,
                                        validation_split=validation_split,
                                        batch_size=4,
                                        data_amount=data_amount,
                                        num_workers=3,
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
    # main()
    make_UA_PCK_curve()