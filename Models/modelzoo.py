import sys
sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

def load_keypointrcnn(load_path,device,num_classes=2,num_keypoints=4,anchor_generator=None):
    anchor_generator = AnchorGenerator(sizes=(128, 256, 512, 1024, 2048), aspect_ratios=(1.0, 2.0, 2.5, 3.0, 4.0))
    
    model = keypointrcnn_resnet50_fpn(weights=None, progress=True, num_classes=num_classes, num_keypoints=num_keypoints,rpn_anchor_generator=anchor_generator)
    model.load_state_dict(torch.load(load_path))
    print(f'Model loaded!')
    model.to(device)
    print(f'Model moved to {device}')
    return model