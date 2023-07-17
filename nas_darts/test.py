import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Network as Network

from create_embedded_dataset import create_embedded_dataset


# parser = argparse.ArgumentParser("text_classify")
# parser.add_argument('--data', type=str, default='data.csv', help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=4, help='batch size')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--init_channels', type=int, default=68, help='num of init channels')
# parser.add_argument('--layers', type=int, default=10, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
# parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
# parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
# parser.add_argument('--seed', type=int, default=0, help='random seed')
# parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
# args = parser.parse_args()

class test_args:
    data = "data.csv"
    batch_size = 8
    report_freq = 50
    gpu = 0
    init_channels = 64
    layers = 8
    model_path = "saved_models/weights.pt"
    auxiliary = False
    drop_path_prob = 0.4
    seed = 5
    arch = "DARTS"
    
args = test_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

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

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    utils.load(model, args.model_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_data, valid_data, test_data = create_embedded_dataset()

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 

