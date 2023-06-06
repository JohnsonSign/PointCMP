# Largely contributed by https://github.com/hehefan/Point-Spatio-Temporal-Convolution

from __future__ import print_function
import datetime
import os
import time
import sys
import random
import numpy as np
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"   
# 4 GPUs 24 batch for linear probing

import torch
import torch.utils.data
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import utils
from logger import setup_logger
from datasets.msr import MSRAction3D
from models.CLR_Model import ContrastiveLearningModel
from timm.loss import LabelSmoothingCrossEntropy


def train(model, criterion, optimizer, lr_scheduler, data_loader, 
            device, epoch, print_freq, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    model.train()

    for i, (clip, target, _) in enumerate(data_loader):
        start_time = time.time()
        clip, target = clip.to(device), target.to(device)
        output = model(clip)
        loss = criterion(output, target)
        batch_size = clip.shape[0]
        lr_ = optimizer.param_groups[-1]["lr"]

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - start_time)

        if i % print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}]\t'
                         'lr: {lr:.5f}\t'
                         'Batch-Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Top1: {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Top5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(data_loader), 
                            lr=lr_, batch_time=batch_time, 
                            loss=losses, top1=top1, top5=top5))) 

    return losses.avg, top1.avg, top5.avg


def evaluate(model, criterion, data_loader, device, print_freq, logger):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for i, (clip, target, video_idx) in enumerate(data_loader):
            start_time = time.time()
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)
            batch_size = clip.shape[0]

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_time.update(time.time() - start_time)

            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            if i % print_freq == 0:
                logger.info(('Test: [{0}/{1}]\t'
                             'Batch-Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Top1: {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Top5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                                i, len(data_loader), batch_time=batch_time, 
                                loss=losses, top1=top1, top5=top5)))

            prob = F.softmax(input=output, dim=1)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k]==video_label[k] for k in video_pred]
    total_acc = torch.tensor(np.mean(pred_correct)).to(device)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
    class_acc = torch.tensor([c/float(s) for c, s in zip(class_correct, class_count)]).to(device)

    logger.info(('Video-level Total-acc: {:.5f}\t'.format(total_acc.item())))
    logger.info(('Video-level Class-acc: {}'.format(np.round(class_acc.tolist(),3))))

    return losses.avg, top1.avg, top5.avg, total_acc.item()


def main(args):

    # Fix the seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    # Check folders and setup logger
    output_dir = os.path.join(args.output_dir, args.model)
    log_dir = os.path.join(args.log_dir, args.model)
    utils.mkdir(output_dir)
    utils.mkdir(log_dir)

    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    logger = setup_logger(output=log_dir, distributed_rank=0, name=args.model)
    tf_writer = SummaryWriter(log_dir=log_dir)

    # Data loading code
    dataset = MSRAction3D(
            root=args.data_path,
            meta=args.data_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            step_between_frames=args.frame_stride,
            num_points=args.num_points,
            sub_clips=args.sub_clips,
            train=True
    )
    train_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    num_workers=args.workers, 
                    shuffle=True,
                    pin_memory=True, 
                    drop_last=True
    )
    dataset_test = MSRAction3D(
            root=args.data_path,
            meta=args.data_meta,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clip_stride,
            step_between_frames=args.frame_stride,
            num_points=args.num_points,
            sub_clips=args.sub_clips,
            train=False
    )
    val_loader = torch.utils.data.DataLoader(
                dataset_test, 
                batch_size=args.batch_size, 
                num_workers=args.workers, 
                pin_memory=True
    )
    # Creat Contrastive Learning Model
    model = ContrastiveLearningModel(
            radius=args.radius, 
            nsamples=args.nsamples, 
            representation_dim=args.representation_dim,
            num_classes=dataset.num_classes,
            pretraining=False
    )
    # Distributed model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    logger.info(("===> Loading checkpoint for finetune '{}'".format(args.finetune)))
    checkpoint = torch.load(args.finetune, map_location='cpu')
    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        if not k.startswith(('module.encoder')):
            del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['module.fc_out.weight', 'module.fc_out.bias']

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if not name.startswith(('module.fc_out')):
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2
    logger.info(("===> Loaded checkpoint with epoch {}".format(checkpoint['epoch'])))

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    optimizer = torch.optim.SGD(parameters, lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay
    )
    warmup_iters = args.lr_warmup_epochs * len(train_loader)
    epochs_iters = args.epochs * len(train_loader)
    lr_scheduler = utils.WarmupCosineLR(optimizer, 
                                        T_max=epochs_iters, 
                                        warmup_iters=warmup_iters, 
                                        last_epoch=-1
    )
    start_time = time.time()
    acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_top1, train_top5 = train(model, criterion, optimizer, 
                                                    lr_scheduler, train_loader, device, 
                                                    epoch, args.print_freq, logger)

        test_loss, test_top1, test_top5, total_acc = evaluate(model, criterion, val_loader, 
                                                            device, args.print_freq, logger)
        acc = max(acc, total_acc)

        logger.info(("Best total acc: '{}'".format(acc)))

        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
        tf_writer.add_scalar('loss/train', train_loss, epoch)
        tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
        tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
        tf_writer.add_scalar('loss/test', test_loss, epoch)
        tf_writer.add_scalar('acc/test_top1', test_top1, epoch)
        tf_writer.add_scalar('acc/test_top5', test_top5, epoch)
        tf_writer.add_scalar('acc/total_acc_best', acc, epoch)
        tf_writer.flush()

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args}
        torch.save(
            checkpoint,
            os.path.join(output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(('Training time {}'.format(total_time_str)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PSTNet Training')

    parser.add_argument('--data-path', default='/data/MSRAction', metavar='DIR', help='path to dataset')
    parser.add_argument('--data-meta', default='datasets/MSRAction_all.list', help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='MSR', type=str, help='model')

    parser.add_argument('--radius', default=0.3, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=9, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--sub-clips', default=4, type=int, metavar='N', help='number of sub-clips')
    parser.add_argument('--clip-stride', default=1, type=int, metavar='N', help='number of steps between clips')
    parser.add_argument('--frame-stride', default=1, type=int, metavar='N', help='number of steps between frames')
    # Following PSTNet, when using a small 'clip-len', increasing 'frame-stride' appropriately will help improve accuracy.
    parser.add_argument('--num-points', default=2048, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('--representation-dim', default=1024, type=int, metavar='N', help='representation dim')

    parser.add_argument('-b', '--batch-size', default=24, type=int) 
    parser.add_argument('--lr', default=0.015, type=float, help='initial learning rate')
    parser.add_argument('--finetune', default='MSR/checkpoint.pth', help='finetune from checkpoint')

    parser.add_argument('--epochs', default=35, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='output/', type=str, help='path where to save')
    parser.add_argument('--log-dir', default='log/', type=str, help='path where to save')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
