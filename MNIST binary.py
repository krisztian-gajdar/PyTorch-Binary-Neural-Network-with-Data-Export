from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from binarized_modules import BinarizeSign, Binarize

# Training settings

parser = argparse.ArgumentParser(description='PyTorch MNIST')

parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--gpus', default=0,
                    help='gpus used for training - e.g 0,1,2')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

mnist_trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
mnist_testset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.htanh1 = nn.Hardtanh()
        self.bin1 = BinarizeSign()
        
        self.fc2 = nn.Linear(2048, 10)
        self.logsoftmax=nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        
        x = self.fc1(x)
        self.fc1.weight.data=Binarize(self.fc1.weight.data)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.bin1(x)
        
        x = self.fc2(x)
        self.fc4.weight.data=Binarize(self.fc2.weight.data)
        return self.logsoftmax(x)


model = Net()
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        data=Binarize(data)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))



def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        data=Binarize(data)
        output = model(data)
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
   train(epoch)
   test()

print("-Start of Data Export")

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.nan)

print("--Start of Running Mean Export")
open('output/mean.txt', 'w').close()
print(model.bn1.running_mean.cpu().detach().numpy(), file=open("output/mean.txt", "a"))
print("--End of Running Mean Export")

print("--Start of Standard Deviation Export")
open('output/std.txt', 'w').close()
print(np.sqrt(model.bn1.running_var.cpu().detach().numpy()), file=open("output/std.txt", "a"))
print("--End of Running Mean Export")

print("--Start of Weight Export")
open('output/weights.txt', 'w').close()
i=0;
for param in model.parameters():
   if i % 2 == 0 :    
       print(param.data.cpu().numpy(), file=open("output/weights.txt", "a"))
   i+=1;
print("--End of Weight Export")

print("--Start of Bias export")
open('output/biases.txt', 'w').close()
i=0;
for param in model.parameters():
   if i % 2 == 1 :    
       print(param.data.cpu().numpy(), file=open("output/biases.txt", "a"))
   i+=1;
print("--End of Bias export")

torch.set_printoptions(profile="default")
print("-End of Data Export")