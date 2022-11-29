import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import Core.torchhelpers.utils as utils
from Core.torchhelpers.coco_eval import CocoEvaluator
from Core.torchhelpers.coco_utils import get_coco_api_from_dataset
from collections import deque
import wandb

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, log_wandb=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # increase window size to be the number of batches, so we can save loss of every step in the MetricLogger
    for loss_type in ["loss","loss_classifier","loss_box_reg","loss_keypoint","loss_objectness","loss_rpn_box_reg"]:
        metric_logger.add_meter(loss_type, utils.SmoothedValue(window_size=len(data_loader)))

    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    print(f'\nRunning training loop!')
    batchnr = 0
    steps_per_epoch = len(data_loader)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        if log_wandb:
            # log metrics to wandb dashboard
            metrics_train = {"train/loss": losses_reduced,
                            "train/loss_classifier":loss_dict_reduced['loss_classifier'],
                            "train/loss_box_reg":loss_dict_reduced['loss_box_reg'],
                            "train/loss_keypoint":loss_dict_reduced['loss_keypoint'],
                            "train/loss_objectness":loss_dict_reduced['loss_objectness'],
                            "train/loss_rpn_box_reg":loss_dict_reduced['loss_rpn_box_reg'],
                            "train/step": steps_per_epoch*epoch+batchnr}
            wandb.log(metrics_train)
        batchnr += 1
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    # added to updata the params.kpt_oks_sigmas array length depending on the number of keypoints in each object in the model
    kp_oks_sigma_init = {"N_kps": data_loader.dataset.dataset.num_keypoints_per_object, "value": .5 / 10}
    # kp_oks_sigma_len = data_loader.dataset.dataset.num_keypoints_per_object
    # oks_sigma_val = .5 / 10
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    print(f'timing function get_coco_api_from_dataset')
    timer_for_fun = time.time()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    print(f'Time of function get_coco_api_from_dataset: {time.time()-timer_for_fun} ({(time.time()-timer_for_fun)/60:.2f} min)')
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, kp_oks_sigma_init)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
