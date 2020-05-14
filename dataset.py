import os
import PIL.Image as Image
import torch
import torchvision
import torchvision.datasets
import torch.utils.data as Data

class JujubeDataset(Data.Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        #构建正样本数据集

        self.imgs0 = os.listdir(os.path.join(img_path,'0'))
        self.imgs1 =os.listdir(os.path.join(img_path,'1'))
        self.imgs=self.imgs0+self.imgs1
        self.lables = [0] * len(self.imgs0)+[1] * len(self.imgs1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            img=Image.open(os.path.join(self.img_path,'0',self.imgs[idx]))
        except:
            img = Image.open(os.path.join(self.img_path, '1', self.imgs[idx]))
        # print(type(img))
        # img=img.resize((512,512))
        if self.transform is not None:
            img = self.transform(img)
        return img,self.lables[idx]


if __name__ == '__main__':
    dataset = JujubeDataset(img_path='data')
    dataset[0]