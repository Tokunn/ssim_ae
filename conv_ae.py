import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchvision.utils import save_image
import torchvision.utils
import os, datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import metrics

import pytorch_ssim

# import visdom
from visdom_utils import VisdomLinePlotter

# Make log dirs
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def choice(tensor0, tensor1, tensor2, n):
    # random.choice
    assert tensor0.size(0)==tensor1.size(0)
    assert tensor1.size(0)==tensor2.size(0)

    perm = torch.randperm(tensor0.size(0))
    idx = perm[:n]
    tensor0 = tensor0[idx]
    tensor1 = tensor1[idx]
    tensor2 = tensor2[idx]
    return tensor0, tensor1, tensor2

def save_diffimage(output, data, truth, filename, epoch,
        filename_roc=None, padding=2, normalize=False, range=None,
        scale_each=False, pad_value=0, max_outputs=30):
    if filename_roc is not None:
        # Calc MSE
        mse = np.asarray([torch.nn.functional.mse_loss(out, da) for (out, da) in zip(output, data)])
        mse = np.reshape(mse, (-1, 5))
        mse = np.mean(mse, axis=1) # TODO SUM?


        # Save ROC
        label = np.asarray([np.round(t.max()) for t in np.reshape(np.asarray(truth), (-1, 5*truth.size(1)*truth.size(2)*truth.size(3)))], dtype=np.int8)
        #predict = [np.mean(p) for p in diff]
        fpr, tpr, threshoulds = metrics.roc_curve(label, mse)
        auc = metrics.auc(fpr, tpr)
        plotter.plot('auc', 'val', 'AUC', epoch, auc)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f'%auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig(filename_roc)

    # Save Diff Image
    torch.manual_seed(10)
    
    output, data, truth = choice(output, data, truth,  max_outputs)
    diff = np.abs(output - data)
    diff = (diff-diff.min())/(diff.max()-diff.min())
    # make grid
    grid_output = torchvision.utils.make_grid(output, nrow=1, padding=padding, pad_value=pad_value,
            normalize=normalize, range=range, scale_each=scale_each)
    grid_data = torchvision.utils.make_grid(data, nrow=1, padding=padding, pad_value=pad_value,
            normalize=normalize, range=range, scale_each=scale_each)
    grid_diff = torchvision.utils.make_grid(diff, nrow=1, padding=padding, pad_value=pad_value,
            normalize=normalize, range=range, scale_each=scale_each)
    grid_truth = torchvision.utils.make_grid(truth, nrow=1, padding=padding, pad_value=pad_value,
            normalize=normalize, range=range, scale_each=scale_each)
    # normalize
    ndarr_output = grid_output.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr_data = grid_data.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr_diff = grid_diff.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr_truth = grid_truth.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # concatenate
    ndarr = np.concatenate([ndarr_data, ndarr_output, ndarr_diff, ndarr_truth], axis=1)

    im = Image.fromarray(ndarr)
    im.save(filename)

class GrayCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, 
            transform=None, target_transform=None, download=False):
        super(GrayCIFAR10, self).__init__(
                root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        # Convert to grayscale
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageFolderRAM(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, resize):
        super(ImageFolderRAM, self).__init__(root, transform)
        self.loaded = [None for _ in range(len(self))]
        self.resize = resize

    def __getitem__(self, index):
        """
        Args:
           index (int): Index
        Returns:
           tuple: (sample, target, rotidx) where target is class_index of the target class.
           rotidx: 0:0, 1:90, 2:180, 3:-90
        """
        path, target = self.samples[index]
        if self.loaded[index]:
            sample = self.loaded[index]
        else:
            sample = self.loader(path)
            sample = self.resize(sample)
            self.loaded[index] = sample

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class MVTechAD(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(MVTechAD, self).__init__(
                root, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        #Convert to grayscale
        sample = sample.convert('L')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #nn.Conv2d(1, 12, 4, stride=2, padding=1),  # b, 16, 10, 10
            #nn.ReLU(True),
            #nn.Conv2d(12, 24, 4, stride=2, padding=1),  # b, 8, 3, 3
            #nn.ReLU(True),
            #nn.Conv2d(24, 48, 4, stride=2, padding=1),  # b, 8, 3, 3

            # Conv1 128x128x1
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ## Conv256 128x128x1
            #nn.Conv2d(32, 32, 4, stride=2, padding=1),
            #nn.ReLU(True),
            # Conv2 64x64x32
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # Conv3 32x32x32
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # Conv4 32x32x32
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Conv5 16x16x64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Conv6 16x16x64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Conv7 8x8x128
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Conv8 8x8x64
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # Conv9 8x8x32
            nn.Conv2d(32, 500, 8, stride=1, padding=0),
            nn.BatchNorm2d(500),
            nn.ReLU(True),
            # 1x1xd
        )
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # b, 16, 5, 5
            #nn.ReLU(True),
            #nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # b, 8, 15, 15
            #nn.ReLU(True),
            #nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),  # b, 1, 28, 28
            #nn.Sigmoid()

            # ConvT9
            nn.ConvTranspose2d(500, 32, 8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # ConvT8
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # ConvT7
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # ConvT6
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # ConvT5
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # ConvT4
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # ConvT3
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # ConvT2
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ## ConvT256
            #nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            #nn.ReLU(True),
            # ConvT1
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            #nn.BatchNorm2d(3),
            nn.Sigmoid()
            #nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = torch.reshape(data, (data.size(0)*5, data.size(2), data.size(3), data.size(4)))
        train_loss = []
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        #if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.sampler),
            100. * batch_idx / len(train_loader), loss.item()))

    # debug save train images
    save_diffimage(output.cpu().data, data.cpu().data, data.cpu().data,
            os.path.join(pngpath, 'train_{}.png'.format(epoch)), epoch)

    train_loss = torch.mean(torch.tensor(train_loss))
    plotter.plot('loss', 'train', LOSS+' Loss', epoch, train_loss)

def test(args, model, criterion, device, test_loader, truth_loader, epoch, now):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for i, ((data, target), (truthdata, truthtarget))in enumerate(
                zip(test_loader, truth_loader)):
            # Open FiveCrop data
            data = torch.reshape(data, (data.size(0)*data.size(1), data.size(2), data.size(3), data.size(4)))
            truthdata = torch.reshape(truthdata, (truthdata.size(0)*truthdata.size(1), truthdata.size(2), truthdata.size(3), truthdata.size(4)))
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss.append(criterion(output, data).item())

    # Save Image
    save_diffimage(output.cpu().data, data.cpu().data, truthdata,
            os.path.join(pngpath, 'image_{}.png'.format(epoch)), epoch,
            os.path.join(pngpath, 'roc_{}.png'.format(epoch)))

    # Show loss
    test_loss = torch.mean(torch.tensor(test_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    plotter.plot('loss', 'val', LOSS+' Loss', epoch, test_loss)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='momentum (default: 0.5)')
    parser.add_argument('--weightdecay', type=float, default=0.00001)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--imgsize', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--loss', type=str, default='SSIM')
    parser.add_argument('--visdom', action='store_true', default=False)
    parser.add_argument('--logname', type=str, default='')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--classes', type=str, default='capsule')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    assert torch.cuda.is_available()==True
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # Loss
    global LOSS
    LOSS = args.loss
    # pngpath
    global pngpath
    pngpath = './dc_img/' + args.logname + now
    os.makedirs(pngpath, exist_ok=True)
    # Visdom
    global plotter
    plotter = VisdomLinePlotter(pngpath, enable=args.visdom)

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    #train_dataset = MVTechAD(os.path.expanduser('~/group/msuzuki/MVTechAD/capsule/train'),
    #train_dataset = GrayCIFAR10('../data', train=True, download=True,
    #train_dataset = ImageFolderRAM(os.path.expanduser('/home/aca10370eo/group/dataset/cifar10/train'),
    train_dataset = ImageFolderRAM(os.path.expanduser('~/group/msuzuki/MVTechAD/{}/train'.format(args.classes)),
                       transform=transforms.Compose([
                           #transforms.RandomResizedCrop(args.imgsize),
                           #transforms.RandomRotation(180),
                           #transforms.Grayscale(),
                           transforms.RandomCrop(args.imgsize),
                           #transforms.FiveCrop(args.imgsize),
                           #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                           #transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           #transforms.Normalize((0.5,), (0.5,))
                       ]),
                       resize = transforms.Resize(args.imgsize*2)
                       )
    train_sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=args.num_samples)

    train_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('../data', train=True, download=True,
        #GrayCIFAR10('../data', train=True, download=True,
                train_dataset, sampler=train_sampler,
                batch_size=args.batch_size, shuffle=False, **kwargs)

        #datasets.MNIST('../data', train=False, transform=transforms.Compose([
        #GrayCIFAR10('../data', train=False, transform=transforms.Compose([
        #MVTechAD(os.path.expanduser('~/group/msuzuki/MVTechAD/capsule/test'), transform=transforms.Compose([
        #ImageFolderRAM(os.path.expanduser('/home/aca10370eo/group/dataset/cifar10/val'), transform=transforms.Compose([
    test_loader = torch.utils.data.DataLoader(
        ImageFolderRAM(os.path.expanduser('~/group/msuzuki/MVTechAD/{}/test'.format(args.classes)), transform=transforms.Compose([
                           #transforms.Resize(args.imgsize),
                           #transforms.Grayscale(),
                           transforms.FiveCrop(args.imgsize),
                           #transforms.ToTensor()
                           transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           #transforms.Normalize((0.5,), (0.5,))
                       ]),
                       resize = transforms.Resize(args.imgsize*2)),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    truth_loader = torch.utils.data.DataLoader(
        ImageFolderRAM(os.path.expanduser('~/group/msuzuki/MVTechAD/{}/ground_truth'.format(args.classes)), transform=transforms.Compose([
                            #transforms.Resize(args.imgsize),
                            transforms.FiveCrop(args.imgsize),
                            #transforms.Grayscale(),
                            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                            #transforms.ToTensor()
                        ]),
                        resize = transforms.Resize(args.imgsize*2)),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Autoencoder()
    #model = nn.DataParallel(model)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True
    #model.apply(weights_init)

    if LOSS=='SSIM':
        print("SSIM Loss")
        criterion = pytorch_ssim.SSIM()
    else:
        print("MSE LOss")
        criterion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    for epoch in range(1, args.epochs + 1):
        print(datetime.datetime.now())
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        test(args, model, criterion, device, test_loader, truth_loader, epoch, now)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

        
if __name__ == '__main__':
    print(torch.cuda.device_count())
    main()

