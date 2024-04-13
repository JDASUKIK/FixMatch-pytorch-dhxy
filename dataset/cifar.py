import logging

import PIL.Image
import math

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

cifar_mean_std= {'cifar10':(cifar10_mean, cifar100_std),
                 'cifar100':(cifar100_mean, cifar100_std)}
def get_cifar10(args, root):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=TransformMultiMatch(args=args, unlabeled=False, dataset='cifar10'))

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformMultiMatch(args=args, unlabeled=True, dataset='cifar10'))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):



    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=TransformMultiMatch(args=args, unlabeled=False, dataset='cifar100'))

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformMultiMatch(args=args, unlabeled=True,dataset='cifar100'))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def denormalization(mean=cifar10_mean, std=cifar10_std, tensor: torch.Tensor=None)->Image.Image:
    # 计算逆标准化的均值和标准差
    inverse_mean = [-m / s for m, s in zip(mean, std)]
    inverse_std = [1 / s for s in std]

    # 构造逆标准化的转换
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=inverse_mean, std=inverse_std),
        transforms.ToPILImage()  # 转换为 PIL 图像格式
    ])(tensor)
    return inverse_normalize

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMultiMatch(object):
    def __init__(self, args, unlabeled=False, dataset='cifar10'):
        self.unlabeled = unlabeled
        mean = cifar_mean_std[dataset][0]
        std = cifar_mean_std[dataset][1]
        crop_class_num = len(args.size_crops)
        if unlabeled == False:
            # labeled image just need simple transform
            self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
            return
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        # weak augmentation: self.weak = list(big1, little1, little2, little3,...)
        self.weak = [transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                args.size_crops[0],
                scale=(args.min_scale_crops[0], args.max_scale_crops[0])),

            self.normalize])] * (args.num_crops[0])
        for crop_cls in range(1, crop_class_num):
            self.weak.extend([transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    args.size_crops[crop_cls],
                    scale=(args.min_scale_crops[crop_cls], args.max_scale_crops[crop_cls])),
                self.normalize])] * (args.num_crops[crop_cls]))
        # strong augmentation: self.strong = list(big1, little1, little2, little3,...)
        self.strong = [transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                args.size_crops[0],
                scale=(args.min_scale_crops[0], args.max_scale_crops[0])),
                RandAugmentMC(n=2, m=10),
            self.normalize] * (args.num_crops[0]))]

        for crop_cls in range(1, crop_class_num):
            self.strong.extend([transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    args.size_crops[crop_cls],
                    scale=(args.min_scale_crops[crop_cls], args.max_scale_crops[crop_cls])),
                    RandAugmentMC(n=2, m=10),
                self.normalize])] * (args.num_crops[crop_cls]))


    def __call__(self, x):
        if self.unlabeled == False:
            return self.transform_labeled(x)
        weak = list(map(lambda trans: trans(x), self.weak))
        strong = list(map(lambda trans: trans(x), self.strong))
        return weak, strong # type:([],[])



class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
