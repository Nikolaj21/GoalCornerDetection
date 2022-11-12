
import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print(f'Running on {device}')
from Core.DataLoader import GoalCalibrationDatasetAUG
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from Core.helpers import eval_PCK,split_data_train_test
import wandb
from utils import DATA_DIR

anchor_generator = AnchorGenerator(sizes=(128, 256, 512, 1024, 2048), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
# remember to update with correct model weights.pth
# load_path = r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection/Models/tester_sgd_da_50epochs/weights.pth'
model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=2, num_keypoints=4,rpn_anchor_generator=anchor_generator)
model.load_state_dict(torch.load(load_path))
model.to(device)
model.eval()
print(f'Model loaded..')

# initialize an instance of the dataloader
GoalData = GoalCalibrationDatasetAUG(DATA_DIR,transforms=None,istrain=False)
GoalData_val = GoalCalibrationDatasetAUG(DATA_DIR,transforms=None,istrain=False)
# put dataloader into pytorch dataloader
_,validation_loader = split_data_train_test(GoalData,GoalData_val,validation_split=0.25,batch_size=4,shuffle_dataset=True,shuffle_seed=None,data_amount=1)

print('Data loaded. Running PCK eval...')

thresholds=[10,30,50,75,100]
PCK = eval_PCK(model,validation_loader,device,thresholds=thresholds)
for pcktype,pck in PCK.items():
    print(f'Percentage of Correct Keypoints (PCK) {pcktype}\n{pck}')
# Log the PCK values in wandb

api = wandb.Api()
# remember to update with correct run path, which can be found under overview section of a run
# run = api.run("nikolaj21/GoalCornerDetection/1k9srkii")

for threshold in thresholds:
    run.summary[f'PCK@{threshold}pix'] = PCK["all_corners"][threshold]
    run.summary[f'PCK_TL@{threshold}pix'] = PCK["top_left"][threshold]
    run.summary[f'PCK_TR@{threshold}pix'] = PCK["top_right"][threshold]
    run.summary[f'PCK_BL@{threshold}pix'] = PCK["bot_left"][threshold]
    run.summary[f'PCK_BR@{threshold}pix'] = PCK["bot_right"][threshold]
run.summary.update()