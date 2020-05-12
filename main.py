import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from resnet import ResNet18
from torchvision.models import resnet18
from log_output import Mylog
from dataset import JujubeDataset
import sys, os

root=os.getcwd()
data_root=os.path.join(root,'data')
log_dir = 'logs/resnet/'

cfg=dict(
    # Hyper Parameters
    EPOCH = 250,        # 训练整批数据多少次, 为了节约时间, 我们只训练一次
    BATCH_SIZE = 4,
    # LR = BATCH_SIZE*0.00125          # 学习率
    LR=0.02,
    lr_scheduler=[130,180,250]
)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
transform_train = transforms.Compose([
            transforms.RandomResizedCrop(512),
            # transforms.RandomRotation(30, resample=Image.BICUBIC, expand=False, center=(256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1),
                                        scale=(0.8, 1.2),
                                        resample=Image.BILINEAR)]),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            normalize,
        ])

trainset=JujubeDataset(img_path='data',transform=transform_train)
#划分数据集
train_full_size=int(len(trainset))
train_size=int(train_full_size*0.8)
val_size=train_full_size-train_size

train_set,valset =torch.utils.data.random_split(trainset,[train_size,val_size])


trainloader =  Data.DataLoader(
    dataset=trainset,
    batch_size=cfg['BATCH_SIZE'],
    shuffle=True,
    num_workers=4
)

val_loader =  Data.DataLoader(
    dataset=trainset,
    batch_size=cfg['BATCH_SIZE'],
    shuffle=True,
    num_workers=4
)
#测试数据集
# testset=torchvision.datasets.CIFAR10(
#     root=data_root,
#     train=False,
#     transform=transform_test
# )
# testloader=Data.DataLoader(
#     dataset=testset,
#     batch_size=256,
#     shuffle=False,
#     num_workers=4
# )
classes = ('good', 'bad')



def data_analyze(dataset):
    #数据分布
    fig1=plt.figure()
    targets=dataset.targets
    plt.hist(targets,bins=10,rwidth=0.8)
    plt.title('dataset histogram')
    plt.xlabel('class_id')
    plt.ylabel('class_num')
    #图片抽样查看
    fig2=plt.figure()
    images=dataset.data[:20]
    for i in np.arange(1,21):
        plt.subplot(5,4,i)
        plt.text(10,10,'{}'.format(targets[i-1]),fontsize=20,color='g')
        plt.imshow(images[i-1])
    fig2.suptitle('Images')
    plt.show()

def adjust_learning_rate(optimizer,lr):
    lr=lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

def save_model(net,filename):
    torch.save(net.state_dict(), filename)

def train(dataloader,net,cfg,optimizer,writer,log=None):
    lr_index=0
    # 随机获取训练图片
    for epoch in range(cfg['EPOCH']):
        Loss=[]
        train_total=0
        train_correct=0
        net.train()
        # read the lr of this epoach
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if epoch>=cfg['lr_scheduler'][lr_index]:
            lr_index+=1
            adjust_learning_rate(optimizer,lr)

        for step,(batch_x,batch_y) in enumerate(trainloader):
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            # print(batch_x.size())
            out=net(batch_x)
            #record train_set acc
            _, predicted = torch.max(out.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            train_acc=train_correct/train_total
            # 计算损失时，直接用非one-hot的展开与准确标签做计算，标签也不是one-hot的，就直接是一个数
            loss=loss_func(out,batch_y)
            optimizer.zero_grad()
            loss.backward()
            Loss.append(loss.data)
            optimizer.step()
            # viz.line([loss.item()], [step], win=train_win, update='append')
            if step%50==0:
                print('epoch:', epoch, 'step:', step, 'lr:{:.5f}'.format(lr),
                      'loss:{:.3f}'.format(loss.data))
        train_loss=sum(Loss)/len(Loss)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Train_acc', train_acc, epoch)
        writer.add_scalar('learning_rate', lr, epoch)
        # lr_sch.step()

        val_acc,val_loss=validate(val_loader,net)
        writer.add_scalar('Test_acc', val_acc, epoch)
        writer.add_scalar('Test_loss', val_loss, epoch)
        if log:
            log.info_out(
                'epoch:{} '.format(epoch)+
                'lr:{:.5f} '.format(lr)+
                'train_acc:{:.3f} '.format(train_acc)+
                'train_loss:{:.3f} '.format(train_loss.data)+
                'val_acc:{:.3f} '.format(val_acc)+
                'val_loss:{:.3f}'.format(val_loss)
            )
        save_model(net,log_dir+'epoach_{}-acc_{}.pth'.format(epoch,val_acc))

def validate(dataloader,net):
    test_Loss = []
    test_total = 0
    test_correct = 0
    # TEST
    with torch.no_grad():
        net.eval()
        for step1, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_x = net(batch_x)
            _, predicted = torch.max(pred_x.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            test_acc = test_correct / test_total
            loss = loss_func(pred_x, batch_y)
            test_Loss.append(loss.data)
        acc=test_acc
        loss=sum(test_Loss) / len(test_Loss)
    return acc,loss


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


if __name__ == '__main__':
    log_dir = 'logs/resnet/'
    # net = densenet121(drop_rate=0.5,num_classes=10)
    net = ResNet18()
    net=resnet18(pretrained=True)

    lr_index=0
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    #
    writer = SummaryWriter(log_dir + TIMESTAMP)
    logger=Mylog(log_dir+'log.txt')
    logger.info_out('>>>>>>>>>>>>>>>>>record>>>>>>>>>>>>>>>>>>>')
    # #绘制网络框图
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg['LR'], momentum=0.9, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    train(trainloader,net,cfg,optimizer,writer,logger)




    # data_analyze(trainset)