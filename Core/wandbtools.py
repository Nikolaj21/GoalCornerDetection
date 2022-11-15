
import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f'Running on {device}')
from Core.DataLoader import GoalCalibrationDatasetAUG
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from Core.helpers import eval_PCK,split_data_train_test
import wandb
from utils import DATA_DIR
import matplotlib.pyplot as plt
import numpy as np

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
    PCK = eval_PCK(model,data_loader,device,thresholds=thresholds)
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



if __name__ == '__main__':
    # main()
    pass


    # for threshold in thresholds:
    #     run.summary[f'PCK@{threshold}pix'] = PCK["all_corners"][threshold]
    #     run.summary[f'PCK_TL@{threshold}pix'] = PCK["top_left"][threshold]
    #     run.summary[f'PCK_TR@{threshold}pix'] = PCK["top_right"][threshold]
    #     run.summary[f'PCK_BL@{threshold}pix'] = PCK["bot_left"][threshold]
    #     run.summary[f'PCK_BR@{threshold}pix'] = PCK["bot_right"][threshold]