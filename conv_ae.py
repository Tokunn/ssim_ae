import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter
import os, datetime
from PIL import Image
import numpy as np

import pytorch_ssim


# Make log dirs
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pngpath = './dc_img/' + now
os.makedirs(pngpath, exist_ok=True)

title = "grayscale AE"
log_dir = './logs/' + title + '/' + now



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
        #img = np.reshape(np.asarray(img), (img.width, img.height, 1))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 4, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # b, 8, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),  # b, 1, 28, 28
            #nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        return x
    
def train(args, model, criterion, device, train_loader, optimizer, epoch):
    #writer = SummaryWriter(log_dir=log_dir)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = criterion(output, data)
        loss = -criterion(output, data) # ssim
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), -loss.item()))

    #writer.add_scalar("train loss", loss.item(), epoch)
    #writer.close()

def test(args, model, criterion, device, test_loader, epoch, now):
    #writer = SummaryWriter(log_dir=log_dir)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, data).item()
            #pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

            save_image(output.cpu().data, './dc_img/{}/image_{}.png'.format(now, epoch), normalize=True)
            # Normalize
            normdata = (output.cpu().data + 1) * 0.5
            #for j in range(5):
                #writer.add_image("test image {}".format((i+1)*5+j), normdata[j], epoch)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #writer.add_scalar("loss test", test_loss)
    #writer.close()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--imgsize', type=int, default=32)
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('../data', train=True, download=True,
        GrayCIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomResizedCrop(args.imgsize),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('../data', train=False, transform=transforms.Compose([
        GrayCIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    ## TensorBoard
    #writer = SummaryWriter(log_dir="logs/" + now)
    ## TB Test
    #x = np.random.randn(100)
    #y = x.cumsum()
    #for i in range(100):
    #    writer.add_scalar("x", x[i], i)
    #    writer.add_scalar("y", y[i], i)
    #writer.close()


    model = Autoencoder().to(device)
    #criterion = nn.MSELoss()
    criterion = pytorch_ssim.SSIM()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        test(args, model, criterion, device, test_loader, epoch, now)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

        
if __name__ == '__main__':
    main()

