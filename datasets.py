from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from augmentation import Augmentation
from torchvision.datasets import ImageFolder
import torchvision.datasets

import numpy as np

# ----------- PlainLoader CIFAR-10 -----------------------


def cifar10(train_bs=128, test_bs=100, train_transform=None, test_transform=None, root='./data', train_val_split_ratio = None, distributed=False):
    if root is None:
        root = "../datasets/cifar10"
    if train_bs is None:
        train_bs = 128
    if test_bs is None:
        test_bs = 100
    train_transform = train_transform or []
    test_transform = test_transform or []
    print("train transform: ", train_transform)
    print("test transform: ", test_transform)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ] + [_get_trans(trans) for trans in train_transform] + [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ] + [_get_trans(trans) for trans in test_transform] + [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if train_val_split_ratio is not None:
 
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
        num_train = len(trainset)
        indices = range(num_train)
        split = int(np.floor(train_val_split_ratio*num_train)) # The percentage fo train in total
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size = train_bs, 
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True,num_workers=4)
        
        validloader = get_inf_iterator(torch.utils.data.DataLoader(
            trainset, batch_size = train_bs, 
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True,num_workers=4
        ))

        ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, pin_memory=True, num_workers=4)

    else:
        trainset = torchvision.datasets.CIFAR10(root= root , train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=8)
        validloader = None
        ori_trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_bs, pin_memory=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root= root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=8)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    return trainloader,validloader,ori_trainloader,testloader, classes

 # * --------------------------------------------------------
from torchvision.datasets import MNIST as MNISTBase


class MNIST(MNISTBase):
    """ MNIST Dataset
    """

    def __init__(self, root, split = 'train', transform = None, label_transform = None, download=True):

        super().__init__(root=root, train = (split == 'train'),
                         transform = transform,
                         download=download)

    @property
    def images(self):
        if self.train:
            return self.train_data
        else:
            return self.test_data

    @property
    def labels(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels


from torchvision.datasets import SVHN as SVHNBase
class SVHN(SVHNBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def images(self):
        return self.data
        
 # * --------------------------------------------------------

from scipy.io import loadmat
from PIL import Image
from torchvision.datasets.utils import download_url
import gzip
import os

class _BaseDataset(Dataset):

    urls          = None
    training_file = None
    test_file     = None
    
    def __init__(self, root, split = 'train', transform = None,
                 label_transform = None, download=True):

        super().__init__()
        
        self.root = root
        self.which = split 
        
        self.transform = transform
        self.label_transform = label_transform

        if download:
            self.download()

        self.get_data(self.which)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        x = Image.fromarray(self.images[index])
        y = int(self.labels[index])
        
        if self.transform is not None:
            x = self.transform(x)

        if self.label_transform is not None:
            y = self.label_tranform(y)
            
        return x, y

    def get_data(self, name):
        """Utility for convenient data loading."""
        if name in ['train', 'unlabeled']:
            self.extract_images_labels(os.path.join(self.root, self.training_file))
        elif name == 'test':
            self.extract_images_labels(os.path.join(self.root, self.test_file))

    def extract_images_labels(self, filename):
        raise NotImplementedError

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok = True)

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root,
                         filename=filename, md5=None)
        # print('Done!')


 # * --------------------------------------------------------

class SynthSmall(_BaseDataset):

    """ Synthetic images dataset
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = {
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32_small.mat?raw=true", 
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32_small.mat?raw=true"
    }
    training_file = 'synth_train_32x32_small.mat?raw=true'
    test_file = 'synth_test_32x32.mat_small?raw=true'
    
    def extract_images_labels(self, filename):
        # print('Extracting', filename)

        mat = loadmat(filename)

        self.images = mat['X'].transpose((3,0,1,2))
        self.labels = mat['y'].squeeze()

class Synth(_BaseDataset):
    """ Synthetic images dataset
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = {
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32.mat?raw=true", 
        "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32.mat?raw=true"
    }
    training_file = 'synth_train_32x32.mat?raw=true'
    test_file = 'synth_test_32x32.mat?raw=true'
    
    def extract_images_labels(self, filename):
        # print('Extracting', filename)

        mat = loadmat(filename)

        self.images = mat['X'].transpose((3,0,1,2))
        self.labels = mat['y'].squeeze()


class USPS(_BaseDataset):
    """
    
    [USPS](http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html) Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.


    Download USPS dataset from [1]_ or use the expliclict links [2]_ for training and [3]_
    for testing.
    Code for loading the dataset partly adapted from [4]_ (Apache License 2.0).

    References: 
        
        .. [1] http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
        .. [2] Training Dataset http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz
        .. [3] Test Dataset http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz
        .. [4] https://github.com/haeusser/learning_by_association/blob/master/semisup/tools/usps.py
    """

    num_labels  = 10
    image_shape = [16, 16, 1]
    
    urls = [
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz'
    ]
    training_file = 'zip.train.gz'
    test_file = 'zip.train.gz'
    
    def extract_images_labels(self, filename):
        import gzip

        # print('Extracting', filename)
        with gzip.open(filename, 'rb') as f:
            raw_data = f.read().split()
        data = np.asarray([raw_data[start:start + 257]
                           for start in range(0, len(raw_data), 257)],
                          dtype=np.float32)
        images_vec = data[:, 1:]
        self.images = np.reshape(images_vec, (images_vec.shape[0], 16, 16))
        self.labels = data[:, 0].astype(int)
        self.images = ((self.images + 1)*128).astype('uint8')

 # * --------------------------------------------------------

class Amazon(ImageFolder):

    def __init__(self,root,split='train',transform = None,download=False):
        super().__init__(root= os.path.join(root,'amazon/images/') ,transform = transform)


class DSLR(ImageFolder):

    def __init__(self,root,split='train',transform = None,download=False):
        super().__init__(root= os.path.join(root, 'dslr/images/') ,transform = transform)

class Webcam(ImageFolder):
 
    def __init__(self,root,split='train',transform = None,download=False):
        super().__init__(root=os.path.join(root, 'webcam/images/'),transform = transform)

class ImagenetSmall(ImageFolder):

    def __init__(self,root,split='train',transform = None,download=False):
        super().__init__(root=os.path.join(root, 'i/'),transform = transform)

class Caltech(ImageFolder):

    def __init__(self,root,split='train',transform = None,download=False):
        super().__init__(root=os.path.join(root, 'c/'),transform = transform)

class Pascal(ImageFolder):

    def __init__(self,root,split='train',transform = None,download=False):
        super().__init__(root=os.path.join(root, 'p/'),transform = transform)




 # * --------------------------------------------------------

from torchvision import transforms
from torch import tensor

def default_normalization(key):

    d = {
    'mnist': (      tensor([ 0.1309,  0.1309,  0.1309]),
                    tensor([ 0.2890,  0.2890,  0.2890])),
    'usps': (       tensor([ 0.1576,  0.1576,  0.1576]),
                    tensor([ 0.2327,  0.2327,  0.2327])),
    'synth':       (tensor([ 0.4717,  0.4729,  0.4749]),
                    tensor([ 0.3002,  0.2990,  0.3008])),
    'synth-small': (tensor([ 0.4717,  0.4729,  0.4749]),
                    tensor([ 0.3002,  0.2990,  0.3008])),
    'svhn':        (tensor([ 0.4377,  0.4438,  0.4728]),
                    tensor([ 0.1923,  0.1953,  0.1904])),
    'amazon':       (tensor([0.485,  0.456,  0.406]),
                    tensor([ 0.229,  0.224,  0.225])),
    'dslr':         (tensor([ 0.485,  0.456,  0.406]),
                    tensor([ 0.229,  0.224,  0.225])),
    'webcam':       (tensor([ 0.485,  0.456,  0.406]),
                    tensor([ 0.229,  0.224,  0.225])),
    'imagenet':     (tensor([0.4561,  0.440,  0.4214]),
                    tensor([ 0.2353,  0.2306,  0.2341])),
    'caltech':      (tensor([ 0.5037,  0.4898,  0.4582]),
                    tensor([ 0.2396,  0.2343,  0.2338])),
    'pascal':       (tensor([ 0.4591,  0.4468,  0.4229]),
                    tensor([ 0.2219,  0.2179,  0.2218]))
    }

    return d[key]

def default_transforms(key):

    d = {

        'mnist' : transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.expand(3,-1,-1).clone())
        ]),

        'svhn' : transforms.Compose([
            transforms.ToTensor(),
        ]),

        'usps' : transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.expand(3,-1,-1).clone())
        ]),

        'synth' : transforms.Compose([
            transforms.ToTensor(),
        ]),

        'synth-small' : transforms.Compose([
            transforms.ToTensor(),
        ]),

        'amazon' : transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),

        'dslr' : transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),

        'webcam' : transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),

        'imagenet' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),

        'caltech' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),

        'pascal' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    
    }

    return d[key]


 # * --------------------------------------------------------

from torch.utils.data import Dataset
from torchvision import datasets, transforms

import h5py
import torch
from torch import nn

import torch.nn.functional as F

import torch.utils.data



class JointLoader:
    # The Joint Loader Works in an infinte Manner
    # If one subset is larger than the other
    # Will Re-initialize the other dataloader
    def __init__(self, *datasets, collate_fn = None):

        self.datasets  = datasets
        self.iterators = [None] * len(datasets)
        self.collate_fn = collate_fn

    def __len__(self):
        # Return the max of the Datasets
        return max([len(d) for d in self.datasets])

    def __iter__(self):
        for i, dataset in enumerate(self.datasets):
            self.iterators[i] = dataset.__iter__()
        return self

    def __next__(self):
        items = []
        for i, dataset in enumerate(self.datasets):
            if self.iterators[i] is None:
                self.iterators[i] = dataset.__iter__()
            try:
                items.append(self.iterators[i].__next__())
            except StopIteration: 
                self.iterators[i] = dataset.__iter__()
                items.append(self.iterators[i].__next__())

        if self.collate_fn is not None:
            items = self.collate_fn(items)

        return items

class MultiDomainLoader(JointLoader):

    """ Wrapper around Joint Loader for multi domain training
    """

    def __init__(self, *args, collate = 'stack'):#, **kwargs):
        assert collate in ['stack', 'cat']

        if collate == 'stack':
            collate_fn = None
        elif collate == 'cat':
            collate_fn = concat_collate
        else:
            raise NotImplementedError

        super().__init__(*args, collate_fn = collate_fn) #, **kwargs)


class DigitsLoader(MultiDomainLoader):
    r""" Digits dataset

    Four domains available: SVHN, MNIST, SYNTH, USPS

    Parameters
    ----------

    root : str
        Root directory where dataset is available or should be downloaded to
    keys : list of str
        pass

    See Also
    --------
    ``torch.utils.data.DataLoader``
    """

    _constructors = {
        'mnist': MNIST,
        'svhn': SVHN,
        'synth': Synth,
        'synth-small': SynthSmall,
        'usps': USPS,
        'amazon': Amazon,
        'dslr': DSLR,
        'webcam': Webcam,
        'imagenet': ImagenetSmall,
        'caltech': Caltech,
        'pascal': Pascal
    }

    def __init__(self, root, keys,
                 split='train', download=True,
                 collate='stack', normalize=False,
                 augment={}, augment_func = Augmentation, batch_size=1,
                 **kwargs):

        assert split in ['train', 'test']

        self.datasets = {}
        for key in keys:
            T = default_transforms(key)
            if normalize:
                # print('Normalize data')
                T.transforms.append(transforms.Normalize(*default_normalization(key)))
            func = self._constructors[key]

            self.datasets[key] = func(root=root, split=split, download=download, transform=T)

            if key in augment.keys():
                self.datasets[key] = augment_func(self.datasets[key], augment[key])

        if isinstance(batch_size, int):
            batch_size = [batch_size] * len(keys)

        super().__init__(*[DataLoader(self.datasets[k], batch_size=b, drop_last=True, **kwargs) for k, b in zip(keys, batch_size)],
                         collate=collate
                         )

# ---------- The Semi-Supervised Scenario ------------------
import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler

# -------------- Transforms ------------------------


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def da_digits(domains = ["svhn","mnist"],train_bs=128, test_bs=100, train_transform=None, test_transform=None, root='./data', label_dir=None, distributed=False,cfg=None):
    if root is None:
        root = "./data"
    if train_bs is None:
        train_bs = 128
    trainloader = DigitsLoader('./data/', domains, shuffle=True, batch_size=train_bs, normalize=True, download=True,num_workers= 1, augment={domains[1]: 2},pin_memory = True)
    testloader = DigitsLoader('./data',domains, shuffle = True, batch_size=train_bs, normalize =True,num_workers = 1,augment_func = None)
    return trainloader, testloader
#



# --------------------------------------------------------------------------------------------------------------
# ---------------- Loader For The Cifar10 Scenario (With Label.txt Denoting Labeled Data))----------------------
# --------------------------------------------------------------------------------------------------------------

NO_LABEL = -1
def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs

# -----------------------------------------------------

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def semi_cifar10(numlabel = 4000,label_bs=64,train_bs=128, test_bs=100, train_transform=None, test_transform=None, root='./data', label_dir=None, distributed=False):
    if root is None:
        root = 'data-local/images/cifar/cifar10/by-image'
    if label_dir is None:
        label_dir = "data-local/labels/cifar10/4000_balanced_labels/00.txt"
    if label_bs is None:
        label_bs = 64
    if train_bs is None:
        train_bs = 128
    if test_bs is None:
        test_bs = 100
    train_transform = train_transform or []
    test_transform = test_transform or []
    print("train transform: ", train_transform)
    print("test transform: ", test_transform)


    transform_train = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dir = os.path.join(root,"train")
    eval_dir = os.path.join(root,"val")

    trainset = torchvision.datasets.ImageFolder(train_dir,transform_train)
    num_train = len(trainset)
    # Pack Up Batch Sampler
    with open(label_dir) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(trainset, labels)
    if numlabel== -1:  # Only Train On Labeled Data
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, train_bs, drop_last=True)
    elif numlabel:   # Train With Both  (Pack The DataSet)
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, train_bs, label_bs)
    else:
        assert False, "labeled batch size {}".format(label_bs)

    # indices = range(num_train)
    # split = int(np.floor(train_val_split_ratio*num_train)) # The percentage fo train in total
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_sampler=batch_sampler,
                                            num_workers= 4,
                                            pin_memory=True)

    testset = torchvision.datasets.ImageFolder(eval_dir, transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=4,pin_memory=True)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    return trainloader ,testloader, classes

# --------------------------------------------------------------------------------------------------------------
# ------------------- Loader For The SVHN Scenario (With Data.npy) ---------------------------------------------
# --------------------------------------------------------------------------------------------------------------

import numpy as np
import os
from PIL import *
import torchvision.transforms as transforms

channel_stats = dict(mean=[0.4377,  0.4438,  0.4728],
                    std=[ 0.1923,  0.1953,  0.1904])

train_transformation = TransformTwice(transforms.Compose([
        # RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([transforms.RandomAffine(90, fillcolor=(100,100,100)) ,transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation=0.5, hue=0.5)], p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))

eval_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

class SVHN_SEMI:
    def __init__(self, root, split=["l_train"], train = True):
        self.dataset = [np.load(os.path.join(root, "svhn", s+".npy"), allow_pickle=True).item() for s in split]
        self.train = train

    def __getitem__(self, idx):
        image = self.dataset[0]["images"][idx]
        label = self.dataset[0]["labels"][idx]
        image_PIL = transforms.ToPILImage()(image.reshape([32,32,3]))  
        if (self.train == True):
            images_aug = train_transformation(image_PIL)
        else:
            images_aug = eval_transformation(image_PIL)
        return images_aug , label

    def __len__(self):
        return sum([len(dataset["images"]) for dataset in self.dataset])

class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    """ Building An Iterator With Num_Iter_Times"""
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def semi_svhn(numlabel = 1000,label_bs=64,train_bs=128, test_bs=100, train_transform=None, test_transform=None, root='./data',distributed=False,cfg=None):
    if root is None:
        root = './data'
    if numlabel != 1000:
        raise Exception("Unsupported Num Label{} , Only 1000 ".format(numlabel))
    if label_bs is None:
        label_bs = 64
    if train_bs is None:
        train_bs = 128
    if test_bs is None:
        test_bs = 100
    train_transform = train_transform or []
    test_transform = test_transform or []
    print("train transform: ", train_transform)
    print("test transform: ", test_transform)

    transform_train = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    # .npy file should be put under folder root+"svhn/$NAME.npy"
    trainset_l = SVHN_SEMI(root,["l_train",],train=True)
    trainset_u = SVHN_SEMI(root,["u_train"],train=True)
    valset = SVHN_SEMI(root,["val"],train=False)
    testset = SVHN_SEMI(root,["test"],train=False)




    # Pack Up Batch Sampler
    trainloader_l = torch.utils.data.DataLoader(trainset_l,
                                                label_bs,
                                                drop_last=True,
                                                sampler=RandomSampler(len(trainset_l),len(trainset_u)*cfg["trainer"]["epochs"]), # the Sampler total should align with the bigger num(The Unlabeled Data)
                                                num_workers= 4,
                                                pin_memory=True)

    trainloader_u = torch.utils.data.DataLoader(trainset_u,
                                                train_bs - label_bs,
                                                drop_last=True,
                                                sampler=RandomSampler(len(trainset_u),len(trainset_u)*cfg["trainer"]["epochs"]),
                                                num_workers= 4,
                                                pin_memory=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=test_bs, shuffle=False, num_workers=4,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=4,pin_memory=True)


    return trainloader_l,trainloader_u,valloader,testloader
