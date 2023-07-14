import os
import sys
import time
import glob
import numpy as np
import torch
from utils import *
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Network as Network

import matplotlib
import matplotlib.pyplot as plt


# parser = argparse.ArgumentParser("text_classify")
# parser.add_argument('--data', type=str, default='data.csv', help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=4, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=68, help='num of init channels')
# parser.add_argument('--layers', type=int, default=10, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
# parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
# parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
# parser.add_argument('--save', type=str, default='EXP', help='experiment name')
# parser.add_argument('--seed', type=int, default=0, help='random seed')
# parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# args = parser.parse_args()

class train_args:
    data = "data.csv"
    batch_size = 8
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    gpu = 0
    epochs = 15
    init_channels = 64
    layers = 8
    model_path = "saved_models"
    auxiliary = False
    auxiliary_weight = 0.4
    drop_path_prob = 0.5
    save = "EXP"
    seed = 5
    arch = "DARTS"
    grad_clip = 5.0
    
args = train_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

NUM_CLASSES = 11


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("%s" % args.arch)
    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_data, valid_data, test_data = create_embedded_dataset()

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
    train_losses = []
    valid_losses = []

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        train_losses.append(train_obj)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        valid_losses.append(valid_obj)
        logging.info('valid_acc %f', valid_acc)

        save(model, os.path.join(args.save, 'weights.pt'))
       
    
    x = [i for i in range(len(train_losses))]
    plt.figure(figsize=(10,5))
    plt.plot(x, train_losses, label="Training Loss", color="b")
    plt.plot(x, valid_losses, label= "Validation Loss", color="r")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.savefig("loss.png")
        
    save(model, os.path.join(args.model_path, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.float()
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.float()
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

main()
