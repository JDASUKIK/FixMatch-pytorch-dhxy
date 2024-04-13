import os
import time
from datetime import datetime

import torch
import random
from torch import nn, optim
from collections import OrderedDict
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler, Subset
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

import utils.logger_setup
from models.mcwideresnet import *
import train
import train_supervised
from models.ema import ModelEMA
from dataset.cifar import *

torch.multiprocessing.set_sharing_strategy('file_system')

best_acc = 0.0

# CIFAR-10 VGG-16 FL

data_root = '/Users/changzhang/PycharmProjects/bupt/DATA/CIFAR-10'


# 返回trainloader和testloder
def getData():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if device == 'cpu':
        trainset = torchvision.datasets.CIFAR10(root='/Users/changzhang/PycharmProjects/bupt/DATA/CIFAR-10/',
                                                train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/Users/changzhang/PycharmProjects/bupt/DATA/CIFAR-10/',
                                               train=False,
                                               download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(root='/home/zc/changzhang/bupt/DATA/CIFAR-10/', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/home/zc/changzhang/bupt/DATA/CIFAR-10/', train=False,
                                               download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 改进VGG网络
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class my_model():
    def __init__(self, lr, epoch_num, num_workers, train_list, test_iter, use_ema: bool):
        self.lr = lr
        self.epoch_num = epoch_num
        self.num_workers = num_workers
        self.net = build_wideresnet(args=args,
                                    depth=args.model_depth,
                                    widen_factor=args.model_width,
                                    dropout=0,
                                    num_classes=args.num_classes).to(device)
        self.ema_model = ModelEMA(args, self.net, args.ema_decay)
        self.train_list = train_list
        self.test_iter = test_iter
        self.use_ema = use_ema

    def getEmaModel(self):
        return self.ema_model

    def getNet(self):
        return self.net

    # 将外来参数加载到网络中
    def load_para(self, public_net_para):
        self.net.load_state_dict(public_net_para)

    # 返回网络的参数字典
    def get_para(self):
        return self.net.state_dict()

    # 返回公共模型参数和本网络参数之差
    def get_minus_para(self, public_net_para, ema=False):
        if self.use_ema:
            para = self.para_minus(self.ema_model.ema.state_dict(), public_net_para)
        else:
            para = self.para_minus(self.net.state_dict(), public_net_para)
        return para

    # 两个模型参数相减
    def para_minus(self, para_a, para_b):
        para = OrderedDict()
        for name in para_a:
            para[name] = para_a[name] - para_b[name]
        return para

    # 使用测试集对模型进行测试
    def test(self, epoch, client):
        correct, total = .0, .0
        for inputs, labels in self.test_iter:
            if device != 'cpu':
                inputs = inputs.cuda()
                labels = labels.cuda()
            self.net.eval()
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Epoch %d, Client %s, Net test accuracy = %.4f %%' % (epoch, client, 100 * float(correct) / total))
        print('-' * 30)

    # 模型训练
    def train(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epoch_num):
            for step, (inputs, labels) in enumerate(iter(self.train_list)):
                if device != 'cpu':
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                output = self.net(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# 返回一个列表，其中包含ceil_num份列表
def list_segmentation(dataloader: DataLoader, ceil_num):
    # 将训练集平均分为x份
    result = []
    length = len(dataloader.dataset)
    step = int(length / ceil_num)
    end_idx = step
    sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    for start_idx in range(0, length - step + 1, step):
        subset = Subset(dataloader.dataset, list(range(start_idx, end_idx)))
        result.append(DataLoader(dataset=subset,
                                 batch_size=dataloader.batch_size,
                                 drop_last=False,
                                 num_workers=dataloader.num_workers,
                                 sampler=sampler(subset)
                                 ))
        end_idx += step
    return result


# 为每个客户端分配一份固定的数据，返回一个字典，key为客户端序号，value为一个存放数据的列表
def set_client_data(client_number, client_data_list):
    result = {}
    for i in range(client_number):
        result[i] = client_data_list[i]
    return result


# 参数列表中所有参数的运算
def para_list_operation(tensor_list, methond='add'):
    assert len(tensor_list) >= 2, '客户端数量低于两个'
    a = tensor_list[0]
    if methond == 'add':
        for i in tensor_list[1:]:
            a += i
    return a / len(tensor_list)


# 对所有参数进行平均
def para_average(para_list, public_net_para):
    net_total_num = len(para_list)
    result = OrderedDict()
    for name in para_list[0]:
        # 2.对所有参数进行平均
        para_sum_list = [para_list[net_num][name] for net_num in range(net_total_num)]
        para_sum = para_list_operation(para_sum_list)
        result[name] = public_net_para[name] + para_sum
    return result


def get_soft_epoch(mean_best_idx):
    if mean_best_idx < 40:
        return 40
    else:
        return mean_best_idx + 10


pub_best = 0


def faderated_train():
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    args.writer = SummaryWriter(logdir)
    # train_loader, test_loader = getData()
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    public_net = my_model(lr=0.003, epoch_num=1, num_workers=2, train_list=[],
                          test_iter=test_loader, use_ema=args.use_ema)

    # args.epochs = math.ceil(args.total_steps / args.eval_step)

    ema_model = public_net.getEmaModel()
    if args.use_ema:
        from models.ema import ModelEMA
        public_net.ema_model = ModelEMA(args, public_net.getNet(), args.ema_decay)
    else:
        ema_model = None
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in public_net.getNet().named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in public_net.getNet().named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(params=grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    scheduler = train.get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    global pub_best
    pub_best = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        # args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        pub_best = checkpoint['best_acc']
        # args.start_epoch = checkpoint['epoch']
        args.comm_round = checkpoint['epoch']
        public_net.getNet().load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            if not checkpoint['ema_state_dict'] == None:
                public_net.ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            else:
                public_net.ema_model.ema.load_state_dict(checkpoint['state_dict'])
            ema_pub_para = checkpoint['ema_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    args.start_epoch = 0

    public_net_para = public_net.get_para()
    ema_pub_para = public_net.getEmaModel().ema.state_dict()
    client_number = 10
    select_number = args.client_num
    communication_rounds = 100
    # 将train_loader分割为client_number份
    labeled_loader_list = list_segmentation(labeled_trainloader, client_number)
    unlabeled_loader_list = list_segmentation(unlabeled_trainloader, client_number)
    # logging
    logger.info(dict(args._get_kwargs()))
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Comment:{comment}")
    logger.info(f"  Num Epochs = {communication_rounds}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    # train.test(args, test_loader, public_net.getNet(), 1, True)
    for epoch in range(args.comm_round, communication_rounds):
        # 每轮训练从客户端随机选取select_number个参与训练
        train_client_list = random.sample(list(range(client_number)), select_number)
        logger.info(f"==>   Train communication:{epoch} begin!")
        logger.info(f"  client list : {str(train_client_list)}, size: {select_number}")

        # print('train_client_list: %s' % train_client_list)
        net_para_list = []
        ema_net_para_list = []
        cli_best_idx_list = []
        cli_best_list = []
        for client in train_client_list:

            logger.info(f"Client {client}:  is training")
            client_net = my_model(lr=args.lr, epoch_num=1, num_workers=args.num_workers
                                  , train_list=[], test_iter=None, use_ema=args.use_ema)
            client_net.load_para(public_net_para)
            client_net.ema_model.ema.load_state_dict(public_net_para)

            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in client_net.getNet().named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
                {'params': [p for n, p in client_net.getNet().named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                                  momentum=0.9, nesterov=args.nesterov)
            scheduler = train.get_cosine_schedule_with_warmup(
                optimizer, args.warmup, args.total_steps)
            train.train(args=args,
                        labeled_trainloader=labeled_loader_list[client],
                        unlabeled_trainloader=unlabeled_loader_list[client],
                        test_loader=test_loader,
                        model=client_net.getNet(),
                        ema_model=client_net.getEmaModel(),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logger=logger
                        )

            # 提前放入一个client的参数，便于更新该list中最后一个参数
            net_para_list.append(client_net.get_minus_para(public_net_para))
            # 网络参数列表增加一项：当前客户端网络与公共网络参数之差
            if args.use_ema:
                ema_net_para_list.append(client_net.get_minus_para(public_net_para, ema=True))

        logger.info(f"parameters training completed, upload and average is in process ...")
        public_net_para = para_average(net_para_list, public_net_para)
        public_net.load_para(public_net_para)
        test_model = public_net.net

        if args.use_ema:
            ema_pub_para = para_average(ema_net_para_list, ema_pub_para)
            public_net.getEmaModel().ema.load_state_dict(ema_pub_para)
            test_model = public_net.getEmaModel().ema

        loss, top1, mask = train.test(args, test_loader=test_loader,
                                      model=test_model, epoch=100, needmask=True)

        logger.info(f"Communication:{epoch}  public-net-> loss:{loss}, top1:{top1}")
        print(f"Communication:{epoch}  public-net-> loss:{loss}, top1:{top1}")
        args.writer.add_scalar('pub_test_loss', loss, epoch)
        args.writer.add_scalar('pub_test_acc', top1, epoch)
        args.writer.add_scalar('pub_mask', mask, epoch)
        if top1 > pub_best:
            pub_best = top1
            if args.use_ema:
                ema_to_save = public_net.getEmaModel().ema.module if hasattr(
                    public_net.getEmaModel().ema, "module") else public_net.getEmaModel().ema
            model_to_save = public_net.net.module if hasattr(public_net.net, "module") else public_net.net
            logger.info(f"ACC:{top1} is current BEST, SAVE in {logdir}")
            train.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': top1,
                'best_acc': pub_best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best=True, checkpoint=logdir, filename=f"pub_model_best_{top1:.2f}.pth.tar")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global args
    args = train_supervised.createArgs()
    nowtime = datetime.strftime(datetime.now(), "%Y-%m-%dT%H.%M.%S")
    global comment
    comment = 'fixmatch+fedavg'
    logdir = f"logs/{args.dataset}@{args.num_labeled}/{comment+'/' if comment != '' else ''}{nowtime}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="%(asctime)s-%(levelname)s:%(message)s",
                        level=logging.DEBUG)
    global logger
    logger = utils.logger_setup.setup_root_logger(logdir, args.local_rank,
                                                  f"size_{args.client_num}_{comment}_train_pub_model.log")
    global client_logger
    client_logger = logging.getLogger("cifar_fl.client")

    if args.local_rank == -1:
        device = torch.device('cuda', int(args.gpu_id))
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device
    args.lr *= args.world_size
    faderated_train()
