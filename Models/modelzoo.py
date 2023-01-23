# import sys
# sys.path.append(r'/zhome/60/1/118435/Master_Thesis/GoalCornerDetection')
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

def loadmymodel(device, anchor_generator, load_path=None, num_keypoints=4, num_classes=2,rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000):
    print(f'Running on {device}')
    model = keypointrcnn_resnet50_fpn(weights=None,
                                      progress=True,
                                      num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      rpn_anchor_generator=anchor_generator,
                                      rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                      rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
    if load_path:
        model.load_state_dict(torch.load(load_path))
        print(f'state_dict loaded from path {load_path}')
        model.eval()
    model.to(device)
    print(f'Model loaded!')
    return model

def load_custom_batch(dataset, im_ids):
    data_plot = [dataset.__getitem__(i) for i in im_ids]
    images,targets = zip(*data_plot)
    return images,targets