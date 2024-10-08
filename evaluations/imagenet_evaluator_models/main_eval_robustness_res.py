import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from evaluations.imagenet_evaluator_models.models_imp import *
from evaluations.imagenet_evaluator_models.data_loader import data_loader, eval_loader_res, val_imagenet_loader_res
from evaluations.imagenet_evaluator_models.helper import AverageMeter, save_checkpoint2, accuracy, adjust_learning_rate

from guided_diffusion import dist_util, logger
from hfai.nn.parallel import DistributedDataParallel as DDP
import hfai
import hfai.multiprocessing
import hfai.nccl.distributed as dist
import hfai.client

from guided_diffusion.script_util import create_classifier

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'unet256na', 'unet64na', 'unet64nas'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--path', required=True ,type=str, help="samples for robustness evaluation")
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--local', action='store_true')
parser.add_argument('--image_size', default=256, type=int)


best_prec1 = 0.0


def main(local_rank):
    global args, best_prec1
    pretrained = True
    args = parser.parse_args()
    diff = False
    # create model
    if pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    t=None
    if args.arch == 'alexnet':
        model = alexnet(pretrained=pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=pretrained)
    elif args.arch == 'unet256na':
        model = create_classifier(image_size=256,
                                  classifier_use_fp16=False,
                                  classifier_width=128,
                                  classifier_depth=2,
                                  classifier_attention_resolutions="32,16,8",
                                  classifier_use_scale_shift_norm=True,
                                  classifier_resblock_updown=True,
                                  classifier_pool="attention",
                                  out_channels=1000,
                                  num_head_channels=64)
        model.load_state_dict(
            dist_util.load_state_dict("models_imp/256x256_classifier.pt", map_location="cpu")
        )

        diff = True
    elif args.arch == 'unet64na':
        model = create_classifier(image_size=64,
                                  classifier_use_fp16=False,
                                  classifier_width=128,
                                  classifier_depth=4,
                                  classifier_attention_resolutions="32,16,8",
                                  classifier_use_scale_shift_norm=True,
                                  classifier_resblock_updown=True,
                                  classifier_pool="attention",
                                  out_channels=1000,
                                  num_head_channels=64)
        model.load_state_dict(
            dist_util.load_state_dict("models_imp/64x64_classifier.pt", map_location="cpu")
        )

        diff = True
    elif args.arch == 'unet64nas':
        model = create_classifier(image_size=64,
                                  classifier_use_fp16=False,
                                  classifier_width=128,
                                  classifier_depth=4,
                                  classifier_attention_resolutions="32,16,8",
                                  classifier_use_scale_shift_norm=True,
                                  classifier_resblock_updown=True,
                                  classifier_pool="attention",
                                  out_channels=1000,
                                  num_head_channels=64)
        model.load_state_dict(
            dist_util.load_state_dict("runs/classifier_training/models/model009999.pt", map_location="cpu")
            # dist_util.load_state_dict("/home/dzung/unisyddev/compt_guidance/runs/latest.pt", map_location="cpu")
        )
        diff = True
    else:
        raise NotImplementedError

    dist_util.setup_dist(local_rank)
    print_rank("_____________________________________________________________________")
    print_rank(f"Eval using {args.arch}")
    print_rank(f"Eval samples: {args.path}")
    print_rank("_____________________________________________________________________")

    # use cuda
    model.to(dist_util.dev())



    dist_util.sync_params(model.parameters())
    # model = torch.nn.parallel.DistributedDataParallel(model)
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        # output_device=dist_util.dev(),
        broadcast_buffers=False,
        # bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # cudnn.benchmark = True

    # Data loading
    mini = args.local
    val_loader = val_imagenet_loader_res(args.batch_size, args.image_size, args.workers, args.pin_memory, mini, diff)
    sample_loader = eval_loader_res(args.batch_size, args.image_size, True, args.path, diff)


    if args.evaluate:
        print_rank("Eval on Imagenet before evaluation on samples")
        best_prec1, _ =  validate(val_loader, model, criterion, args.print_freq, -1, diff)
        print_rank("Start evaluation")

    validate(sample_loader, model, criterion, args.print_freq, -1, diff)



def validate(val_loader, model, criterion, print_freq, epoch, diffuse=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_acc, correct1, correct5, total = torch.zeros(4).cuda()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        if diffuse:
            t = torch.zeros(target.shape[0], dtype=torch.long, device=dist_util.dev())
        else:
            t = None
        with torch.no_grad():
            # compute output
            if t is None:
                output = model(input)
            else:
                output = model(input, timesteps=t)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print_rank('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
    batch_num = i + 1
    loss_acc += losses.sum
    correct1 += (top1.sum / 100.0)
    correct5 += (top5.sum / 100.0)
    total += top1.count
    for x in [loss_acc, correct1, correct5, total]:
        dist.reduce(x, 0)
    if dist.get_rank() == 0:
        loss_val = loss_acc.item() / dist.get_world_size() / batch_num
        acc1 = 100.0 * correct1.item() / total.item()
        acc5 = 100.0 * correct5.item() / total.item()
        print(f'Epoch: {epoch}, Loss: {loss_val}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return correct1.item()/total.item(), correct5.item()/total.item()


def print_rank(message: str, main_rank=0):
    if dist.get_rank() == main_rank:
        print(message)



if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
