import argparse
import sys
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader import ImagerLoader
import numpy as np
from videogaze_model import VideoGaze,CompressedModel
import cv2
import math
import sklearn.metrics
from config import *
from datetime import datetime
from sklearn import metrics
from scipy import stats
import pdb

from train import *


# TACC jobs don't like stdout. Print to stderr
def eprint(*args, **kwargs):
    print(str(datetime.now().strftime('%H:%M:%S')),":", *args, file=sys.stderr, **kwargs)

def transformToMap(ground_truth_array):
    grid = np.zeros([batch_size, side_w, side_w])
    indexedArray = ground_truth_array * side_w
    indexedArray = indexedArray.floor().int()
    for i in range(len(indexedArray)):
        if indexedArray[i][0][0] > 0 and indexedArray[i][0][0] > 0:
            grid[i][indexedArray[i][0][0]][indexedArray[i][0][1]] = 1
    return grid

def validate_stats(val_loader, model, criterion,criterion_b):
    global count_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    topb = AverageMeter()
    l2 = AverageMeter()
    kl = AverageMeter()
    auc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (source_frame,target_frame,face_frame,eyes,target_i,gaze_float,binary_tensor) in enumerate(val_loader):
        #Convert target values into grid format
        target = float2grid(target_i,side_w)
        target = target.float()

        # Prepare tensors and variables
        source_frame = source_frame.cuda(async=True)
        target_frame = target_frame.cuda(async=True)
        face_frame = face_frame.cuda(async=True)
        eyes = eyes.cuda(async=True)
        target = target.cuda(async=True)
        binary_tensor = binary_tensor.cuda(async=True)
        target_i = target_i.cuda(async=True)


        source_frame_var = torch.autograd.Variable(source_frame)
        target_frame_var = torch.autograd.Variable(target_frame)
        face_frame_var = torch.autograd.Variable(face_frame)
        eyes_var = torch.autograd.Variable(eyes)
        target_var = torch.autograd.Variable(target)
        binary_var = torch.autograd.Variable(binary_tensor.view(-1))

        # compute output
        output,sigmoid= model(source_frame_var,target_frame_var,face_frame_var,eyes_var)
        # output,sigmoid= model(source_frame_var,target_frame_var)

        ground_truth_gaze = transformToMap(gaze_float)
        prediction_grid = output.data.cpu().numpy()
        to_skip = False
        for b in range(prediction_grid.shape[0]):
            # auc is undefined if ground_truth is all zeros
            if np.any(ground_truth_gaze[b]):
                cur_auc = metrics.roc_auc_score(ground_truth_gaze[b].flatten(),
                        prediction_grid[b].flatten())
                auc.update(cur_auc)

            # KL divergence isn't undefined if ground_truth is all zeros
            prediction_grid[b][prediction_grid[b] == 0] = 1e-6
            ground_truth_gaze[b][ground_truth_gaze[b] == 0] = 1e-6
            cur_kl = stats.entropy(prediction_grid[b].flatten(), ground_truth_gaze[b].flatten())
            kl.update(cur_kl)

        #Compute loss
        loss_l2 = criterion(output, target_var)
        loss_b = criterion_b(sigmoid, binary_var)
        loss = loss_l2+12*loss_b

        # measure performance and record loss
        prec1, prec5 = accuracy(output.data, target_i.view(-1), topk=(1, 5))
        prec1_b = ap_b(sigmoid.data, binary_tensor, topk=(1,))
        l2e = l2_error(output, target_i.view(-1),gaze_float)
        losses.update(loss.data[0], source_frame.size(0))
        top1.update(prec1[0], source_frame.size(0))
        top5.update(prec5[0], source_frame.size(0))
        l2.update(l2e, source_frame.size(0))
        topb.update(prec1_b, source_frame.size(0))


        batch_time.update(time.time() - end)
        end = time.time()


        eprint('Batch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \t'
                  'Binary {topb.val:.3f} ({topb.avg:.3f}) \t'
                  'L2 {l2.val:.3f} ({l2.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5,l2=l2,topb=topb))
        eprint('AUC: {auc.val:.3f} ({auc.avg:.3f}) \t'
                  'KL: {kl.val:.3f} ({kl.avg:.3f})'.format(
                    auc=auc, kl=kl))

    return l2.avg




def main():
    global args, best_prec1,weight_decay,momentum

    model = VideoGaze(bs=batch_size,side=side_w)
    checkpoint = torch.load('checkpoint_short.pth.tar')

    # model = CompressedModel(bs=batch_size,side=side_w)
    # checkpoint = torch.load('checkpoint_compressed_short_combined.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    cudnn.benchmark = True

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,face_path,target_path,test_file,transforms.Compose([
            transforms.ToTensor(),
        ]),square=(227,227),side=side_w),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    criterion = ExponentialShiftedGrids().cuda()
    criterion_b = nn.BCELoss().cuda()

    # evaluate on validation set
    prec1 = validate_stats(val_loader, model, criterion,criterion_b)
    print(prec1)

if __name__ == '__main__':
    main()

