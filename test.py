import os
print(f'current path: {os.getcwd()}')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def terminaltest():
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}')
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_generator = AnchorGenerator(sizes=(128, 256, 512, 1024, 2048), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    # remember to update with correct model weights.pth
    load_path = r'/zhome/60/1/118435/Master_Thesis/Runs/tester_sgd_da_50epochs/weights.pth'
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=2, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.load_state_dict(torch.load(load_path))
    model.to(device)
    return model

def testPCK():
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Running on {device}')
    from Core.DataLoader import GoalCalibrationDataset4boxes, GoalCalibrationDataset, train_transform
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.rpn import AnchorGenerator
    from Core.helpers import eval_PCK,split_data_train_test
    from utils import DATA_DIR
    import matplotlib.pyplot as plt
    import time
    import numpy as np
    anchor_generator = AnchorGenerator(sizes=(64, 128, 256, 512, 1024), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    # remember to update with correct model weights.pth
    load_path = r"/zhome/60/1/118435/Master_Thesis/Scratch/s163848/Runs/1box_check_5epochs/weights.pth"
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=2, num_keypoints=4,rpn_anchor_generator=anchor_generator)
    model.load_state_dict(torch.load(load_path))
    model.to(device)
    model.eval()
    print(f'Model loaded..')

    # initialize an instance of the dataloader
    GoalData = GoalCalibrationDataset(DATA_DIR,transforms=None)
    GoalData_val = GoalCalibrationDataset(DATA_DIR,transforms=None)
    # put dataloader into pytorch dataloader
    _,validation_loader = split_data_train_test(
                                                GoalData,
                                                GoalData_val,
                                                validation_split=0.25,
                                                batch_size=1,
                                                data_amount=1,
                                                num_workers=0,
                                                shuffle_dataset=True,
                                                shuffle_dataset_seed=10,
                                                shuffle_epoch=False,
                                                shuffle_epoch_seed=-1)

    print('\nData loaded. Running PCK eval...')

    import numpy as np
    thresholds = np.arange(1,101)
    PCK,pixelerrors = eval_PCK(model,validation_loader,device,thresholds=thresholds,num_objects=1)

    for pck_type,pck_values in PCK.items():
        pck_values = list(pck_values.values())
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(thresholds,pck_values)
        ax.set_title(f'PCK curve for type: {pck_type}')
        ax.set_xlabel('threshold')
        ax.set_ylabel('PCK')
        fig.savefig(f'PCK_{pck_type}.png')

    print('We are DONE :)')

def test_dataloader_speed():
    import torch
    import sys
    # sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Running on {device}')
    from Core.DataLoader import GoalCalibrationDataset4boxes, GoalCalibrationDataset, train_transform
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.rpn import AnchorGenerator
    from Core.helpers import eval_PCK,split_data_train_test, test_num_workers
    from utils import DATA_DIR
    import matplotlib.pyplot as plt
    import time
    import numpy as np
    num_workers = 2
    print(f'num_workers: {num_workers}')
    GoalData = GoalCalibrationDataset4boxes(DATA_DIR)
    GoalData_da = GoalCalibrationDataset4boxes(DATA_DIR,transforms=train_transform())
    # print('\n###################################################################')
    # print(f'Testing number of workers for dataset without data augmentation\n')
    # test_num_workers(GoalData,batch_size=8,data_amount=0.75)
    # print('\n###################################################################')
    # print(f'Testing number of workers for dataset WITH data augmentation\n')
    # test_num_workers(GoalData_da,batch_size=8,data_amount=0.75)
    train_loader,validation_loader = split_data_train_test(
                                                            GoalData,
                                                            GoalData,
                                                            validation_split=0.25,
                                                            batch_size=8,
                                                            data_amount=1,
                                                            num_workers=num_workers,
                                                            shuffle_dataset=True,
                                                            shuffle_dataset_seed=20,
                                                            shuffle_epoch = False,
                                                            shuffle_epoch_seed=-1,
                                                            pin_memory=False)
    train_loader_pin,validation_loader_pin = split_data_train_test(
                                                            GoalData,
                                                            GoalData,
                                                            validation_split=0.25,
                                                            batch_size=8,
                                                            data_amount=1,
                                                            num_workers=num_workers,
                                                            shuffle_dataset=True,
                                                            shuffle_dataset_seed=20,
                                                            shuffle_epoch = False,
                                                            shuffle_epoch_seed=-1,
                                                            pin_memory=True)
    n_runs = 3
    times = []
    # simulate going through all the data and moving to device, run multiple times and average runtimes
    print(f'Testing time for moving all data to device with pin_memory=False...')
    for i in range(n_runs):
        start = time.time()
        for images,targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        runtime = time.time() - start
        times.append(runtime)
        print(f'Time for run {i}: {runtime} s ({runtime/60:.2f} min)')
    print(f'Average time pin_memory=False: {np.mean(times)} s ({np.mean(times)/60:.2f} min)')
    
    times = []
    # simulate going through all the data and moving to device, run multiple times and average runtimes
    print(f'Testing time for moving all data to device with pin_memory=True...')
    for _ in range(n_runs):
        start = time.time()
        for images,targets in train_loader_pin:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        runtime = time.time() - start
        times.append(runtime)
        print(f'Time for run {i}: {runtime} s ({runtime/60:.2f} min)')
    print(f'Average time pin_memory=True: {np.mean(times)} s ({np.mean(times)/60:.2f} min)')

def test_paths():
    import os
    from pathlib import Path,PurePath
    import glob
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(ROOT_DIR,'Data')
    test_dir = Path(__file__)
    print(f'DATA_DIR: {DATA_DIR}')
    print(f'test_dir: {test_dir}')
    r_dir = PurePath(ROOT_DIR)
    print(f'r_dir: {r_dir}')
    datapath = 'C:\\Users\\Nikolaj\\OneDrive - Danmarks Tekniske Universitet\\DTU\\Kandidat\\MasterThesis\\Code\\GoalCornerDetection\\Data'
    data_purepath = PurePath(datapath)
    print(f"wdd joined with crap: {os.path.join(datapath,'/*/*.jpg')}")
    print(f"wdd_path joined with crap: {os.path.join(data_purepath,PurePath('/*/*.jpg'))}")
    print(f"wdd pathed with crap: {Path(datapath+'/*/*.jpg')}")

    # img_list = sorted(glob.glob(str(Path(datapath + '/*/*.jpg'))))
    # glob_path = str(Path( str(Path(datapath)) + str(Path('/*/*.jpg')) ))
    glob_path = str(Path(datapath+'/*/*.jpg'))
    
    print(f'glob_path: {glob_path}')
    img_list = sorted(glob.glob(glob_path))
    print(f'img_list items:\n{img_list[:3]}')

from utils import DATA_DIR
class Params:
    def defaultparams(self):
        self.data_dir = DATA_DIR
        self.batch_size = 4
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
        self.model_name = "tester_model"
        self.pckthreshup = 200
        self.predims_every = 5
        self.data_amount = 1
        self.shuffle_dataset = "True"
        self.shuffle_epoch = "False"
        self.shuffle_dataset_seed = -1
        self.shuffle_epoch_seed = -1
        self.model_type = "4box"
        self.filter_data = "True"

        self.test_only = False
        self.load_path = None
        self.data_aug = False


    def setself(self, params):
        for name,val in params.items():
            self.__dict__[name] = val
    def __init__(self):
        self.defaultparams()

params = Params()

def test_wandb_init(args=None):
    import wandb
    import argparse
    from distutils.util import strtobool
    wandb.init(config=None)
    wandb.config.epochs = 4

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument("--filter-data", default='True', type=bool(strtobool), nargs='?', const=False, help="Shuffle which data ends up in train set and validation set. Default: True")

    args = parser.parse_args()
    wandb.config.update(args) # adds all of the arguments as config variables
    print(wandb.config)

def main():
    # testPCK()
    # test_dataloader_speed()
    # test_paths()
    test_wandb_init()

if __name__ == "__main__":
    main()

