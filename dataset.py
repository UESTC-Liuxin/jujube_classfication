import os
import PIL.Image as Image
import torch
import torchvision
import torchvision.datasets
from bbox_extract import *
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
            file_name=os.path.join(self.img_path,'0',self.imgs[idx])
            (x,y,w,h),img=draw_bbox(file_name)
            # cv2.imshow('flip',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except:
            file_name=os.path.join(self.img_path,'1',self.imgs[idx])
            (x,y,w,h),img=draw_bbox(file_name)
        img = img[y - 50:y + h+50, x - 50:x + w + 50, :]
        pil_img = Image.fromarray(img)
        # print(type(img))
        # cv2.imshow('flip',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pil_img=pil_img.resize((512,512))
        if self.transform is not None:
            pil_img = self.transform(pil_img)
        return pil_img,self.lables[idx]


if __name__ == '__main__':
    dataset = JujubeDataset(img_path='data')
    for i in range(100):
        dataset[i]
