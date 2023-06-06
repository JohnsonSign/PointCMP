from __future__ import print_function
import datetime
import os
import time
import sys
import random
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.utils.data
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import utils
from logger import setup_logger
from data_aug.CLR_MSR import CLRMSRSubject
from models.CLR_Model import ContrastiveLearningModel
from timm.scheduler import CosineLRScheduler


def train(model, optimizer, lr_scheduler, data_loader, 
        device, epoch, print_freq, logger, criterion_global):

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.train()
    for i, (clips, index) in enumerate(data_loader):
        start_time = time.time()
        
        clips = clips.to(device)  # [B, S, L, N, 3]

        loss, acc1, acc5, view1_global, erase_global, regression_global = model(clips)
        loss, acc1, acc5 = loss.mean(), acc1.mean(0), acc5.mean(0)

        view11 = torch.matmul(view1_global, regression_global.transpose(0,1)) # [B, B]

        view12 = view1_global.unsqueeze(1) * erase_global                     # [B, H, C]
        view12 = torch.sum(view12, dim=-1, keepdim=False)                     # [B, H]

        view_score = torch.cat(tensors=(view11, view12), dim=1)               # [B, B+H]
        view_score = view_score / 0.1

        target_view_score = torch.arange(view_score.shape[0]).to(device)
        loss_global = criterion_global(view_score, target_view_score)

        loss = loss + loss_global 

        batch_size = clips.shape[0]
        lr_ = optimizer.param_groups[-1]["lr"]

        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - start_time)

        if i % print_freq == 0:
            logger.info(('Epoch:[{0}][{1}/{2}]\t'
                         'lr:{lr:.5f}\t'
                         'Loss:{loss.val:.3f} ({loss.avg:.3f})\t'
                         'Top1:{top1.val:.2f} ({top1.avg:.2f})'.format(
                          epoch, i, len(data_loader), lr=lr_, loss=losses, top1=top1))) 

    return losses.avg, top1.avg, top5.avg


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
    log_dir = os.path.join(args.log_dir, args.model)
    utils.mkdir(log_dir)

    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    logger = setup_logger(output=log_dir, distributed_rank=0, name=args.model)
    tf_writer = SummaryWriter(log_dir=log_dir)

    # Data loading code
    train_dataset = CLRMSRSubject(
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
                    train_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    num_workers=args.workers, 
                    pin_memory=True, 
                    drop_last=True
    )
    # Creat Contrastive Learning Model
    model = ContrastiveLearningModel(
            radius=args.radius, 
            nsamples=args.nsamples, 
            representation_dim=args.representation_dim,
            temperature=args.temperature,
            pretraining=True
    )
    # Distributed model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion_global = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay
    # )
    # param_groups = utils.add_weight_decay(model, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # lr_scheduler = CosineLRScheduler(optimizer,
    #         t_initial=args.epochs, t_mul=1, lr_min=1e-6, decay_rate=0.1,
    #         warmup_lr_init=1e-6, warmup_t=args.lr_warmup_epochs, cycle_limit=1, t_in_epochs=True
    # )
    warmup_iters = args.lr_warmup_epochs * len(train_loader)
    epochs_iters = args.epochs * len(train_loader)
    lr_scheduler = utils.WarmupCosineLR(optimizer, 
                                        T_max=epochs_iters, 
                                        warmup_iters=warmup_iters, 
                                        last_epoch=-1
    )

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("===> Loading checkpoint for resume '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            logger.info(("===> Loaded checkpoint with epoch {}".format(checkpoint['epoch'])))
        else:
            logger.info(("===> There is no checkpoint at '{}'".format(args.resume)))

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_top1, train_top5 = train(model, optimizer, lr_scheduler, train_loader, device, 
                                                    epoch, args.print_freq, logger, criterion_global
        )
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
        tf_writer.add_scalar('loss/train', train_loss, epoch)
        tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
        tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
        tf_writer.flush()

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        torch.save(
            checkpoint,
            os.path.join(log_dir, 'checkpoint_{}.pth'.format(epoch))
        )
        logger.info('====================================')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(('Training time {}'.format(total_time_str)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PointCMP')
    parser.add_argument('--data-path', default='/data/MSRAction', metavar='DIR', help='path to dataset')
    parser.add_argument('--data-meta', default='datasets/MSRAction_all.list', help='dataset')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=96, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--temperature', default=0.01, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--representation-dim', default=1024, type=int, metavar='N', help='representation dim')

    parser.add_argument('--sub-clips', default=4, type=int, metavar='N', help='number of sub clips')
    parser.add_argument('--radius', default=0.3, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=9, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--clip-stride', default=1, type=int, metavar='N', help='number of steps between clips')
    parser.add_argument('--frame-stride', default=1, type=int, metavar='N', help='number of steps between clips')
    parser.add_argument('--num-points', default=1024, type=int, metavar='N', help='number of points per frame')

    parser.add_argument('--model', default='MSR', type=str, help='model')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--print-freq', default=200, type=int, help='Log every n steps')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--log-dir', default='log_ssl/', type=str, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
