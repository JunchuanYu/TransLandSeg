from re import M
from PIL import ImageFilter
import random
import h5py
# import gdal
# from gdalconst import *
# from osgeo import gdal_array
import numpy as np
import torch
import torchvision
from PIL import Image
import os
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torchvision.transforms import InterpolationMode

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting two transformations from one data
    Args:
        images_dir : path of input images
        augmentation : MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    Output:
        q : an augmatation for input 
        k : annother different augmention for the same input"""

    def __init__(self, images_dir,  augmentation=None):
        self.images = [i for i in os.listdir(images_dir) if i.endswith('.tif')]
        self.images_dir = images_dir
        self.augmentation = augmentation

        if self.augmentation == None:
            self.augmentation = [
                torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                # torchvision.transforms.RandomApply([
                #     torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                #     ], p=0.8),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.1121, 0.1091, 0.1000],
                                        # std=[0.0333, 0.0139, 0.0074])
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
                
                ]
      
        self.tx = torchvision.transforms.Compose(self.augmentation)


    def __len__(self):

        return len(self.images)


    def __getitem__(self, i):
        img_file = gdal.Open(self.images_dir + self.images[i], GA_ReadOnly)
        img_arr = img_file.ReadAsArray()

        img1 = np.array(img_arr) # (C, H, W)
        img1 = np.transpose(img1, (1, 2, 0))
        
        PILimg = Image.fromarray(img1)# 0~255

        q = self.tx(PILimg)
        k = self.tx(PILimg)

        return q, k

class h5_Dataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
    Output:
        img = c,h,w
        label = c,h,w"""

    def __init__(self, images_dir, labels_dir):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir


    def __len__(self):

        return len(self.images)


    def from_14chanel_to_3chanel(self, array, R=4, G=3, B=2):
        """
        从一个14通道的影像取出3个通道通道影像
        array: chanel last
        """
        return np.concatenate((array[:,:,[R-1]], array[:,:,[G-1]], array[:,:, [B-1]]), axis=2)


    def __getitem__(self, i):
        img_h5 = h5py.File(self.images_dir + self.images[i], 'r')
        label_h5 = h5py.File(self.labels_dir + self.labels[i], 'r')

        img1 = np.array(img_h5['img'])
        img1 = self.from_14chanel_to_3chanel(img1)
        
        PILimg = Image.fromarray((img1 / 10 * 255).astype(np.uint8))# 0~255

        label1 = np.array(label_h5['mask']).reshape(-1, 128, 128)

        img = torch.from_numpy(np.array(PILimg).transpose(2, 0, 1))/255
        label = torch.from_numpy(label1)

        return img, label

class tif_Dataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
    Output:
        img = c,h,w
        label = c,h,w"""

    def __init__(self, images_dir, labels_dir, transform_I=None, transform_L=None):
              
        img_list = sorted(os.listdir(images_dir))
        self.images = [i for i in img_list if i.endswith('.tif')]

        label_list = sorted(os.listdir(labels_dir))
        self.labels = [i for i in label_list if i.endswith('.png')]
        
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
        # self.count = 0
        # self.images = []
        # for i in os.listdir(images_dir):
        #     if i.endswith('.tif'):
        #         self.count = self.count + 1
        #         self.images(i)

        
        
        if transform_I:
            self.transform_I = transform_I
        else:
            self.transform_I = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((512,512)),
                torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.08, 0.5), ratio=(1, 1)),
                # torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)], p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if transform_L:
            self.transform_L = transform_L
        else:
            self.transform_L = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((512,512)),
                torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.08, 0.5), ratio=(1, 1), interpolation=Image.NEAREST),
                # torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
            ])


    def __len__(self):

        return len(self.images)


    def Load_image_by_Gdal(self, file_path):
        img_file = gdal.Open(file_path, GA_ReadOnly)
        # if img_file is None:
        #     raise 'ERROR: fail to open TIF file'
        img_arr = img_file.ReadAsArray()  # 获取投影信息
        if img_arr.ndim == 2:
            img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = img_arr.transpose((1, 2, 0)) # 转为H,W,C  
        return img_arr



    def __getitem__(self, i):
        img_arry = self.Load_image_by_Gdal(self.images_dir + self.images[i])
        # label_arry = self.Load_image_by_Gdal(self.labels_dir + self.labels[i])
        PILlabel = Image.open(self.labels_dir + self.labels[i])
        
        PILimg = Image.fromarray(img_arry.astype(np.uint8))# 0~255
        # PILlabel = Image.fromarray(label_arry.astype(np.uint8)) # 0~255

        seed=np.random.randint(0, int(2**31)) # make a seed with numpy generator 

        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.transform_I(PILimg)
        
        # apply this seed to target/label tranfsorms  
        random.seed(seed) 
        torch.manual_seed(seed)
        label = self.transform_L(PILlabel)

        return img, label
    
class png_Dataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
    Output:
        img = c,h,w
        label = c,h,w"""

    def __init__(self, images_dir, labels_dir, transform_I=None, transform_L=None):
              
        img_list = sorted(os.listdir(images_dir))
        
        self.images = [i for i in img_list if i.endswith('.png')]
        
        label_list = sorted(os.listdir(labels_dir))

        self.labels = [i for i in label_list if i.endswith('.png')]
       
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.inp_size=1024
        # self.count = 0
        # self.images = []
        # for i in os.listdir(images_dir):
        #     if i.endswith('.tif'):
        #         self.count = self.count + 1
        #         self.images(i)
        self.img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((self.inp_size, self.inp_size)),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
        self.mask_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((self.inp_size, self.inp_size)),
        torchvision.transforms.ToTensor(),
    ])
        



    def __len__(self):

        return len(self.images)


    def __getitem__(self, i):
        # print(self.images_dir + self.images[i])
        img_arry = Image.open(self.images_dir + self.images[i])
        
        # label_arry = self.Load_image_by_Gdal(self.labels_dir + self.labels[i])
        # print(self.labels_dir + self.labels[i])
        PILlabel = Image.open(self.labels_dir + self.labels[i])
        
        # image_array = np.array(PILlabel)
        # image_array *= 255
        # PILlabel = Image.fromarray(image_array.astype(np.uint8))
        # PILimg = Image.fromarray(img_arry.astype(np.uint8))# 0~255
        # PILlabel = Image.fromarray(label_arry.astype(np.uint8)) # 0~255

        # seed=np.random.randint(0, int(2**31)) # make a seed with numpy generator 

        # # apply this seed to img tranfsorms
        # random.seed(seed) 
        # torch.manual_seed(seed)
        # img = self.transform_I(PILimg)
        
        # # apply this seed to target/label tranfsorms  
        # random.seed(seed) 
        # torch.manual_seed(seed)
        # label = self.transform_L(PILlabel)
        if random.random() < 0.5:
            img =  img_arry.transpose(Image.FLIP_LEFT_RIGHT)
            mask = PILlabel.transpose(Image.FLIP_LEFT_RIGHT)

        img = torchvision.transforms.Resize((self.inp_size, self.inp_size))(img_arry)
        mask = torchvision.transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(PILlabel)

        return self.img_transform(img),torch.where(self.mask_transform(mask) > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        
   
    
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     def plot_func(data_list, n=2, camp='gray', mean=None, std=None):
#         '''
#         data: list of numpy array or tensor
#         n: number of dataset in data_list
#         '''
#         for m in range(n):
#             fig=plt.figure(figsize=(30,8))
#             for i in range(9):
#                 plt.subplot(1,9,i+1)
#                 if 'cuda' in data_list[m].device.type:
#                     img = data_list[m][i,:,:,:].cpu().numpy().transpose(1, 2, 0)
#                 else:
#                     img = data_list[m][i,:,:,:].numpy().transpose(1, 2, 0)
#                 if camp == 'gray' and img.shape[-1] == 1:
#                     plt.imshow(img, cmap=camp)
#                 elif mean and std and img.shape[-1] == 3:
#                     plt.imshow(img *std + mean)
#                 else:
#                     plt.imshow(img)
#             plt.show()
#     # dataset = Images_Dataset_folder('Z:\\xuxin\\Tibet_lake\\data\\negative_data\\img\\')
#     # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0)
#     # for q, k in train_loader:
#     #     print(q.shape), print(k.shape)
#     #     break
#     # print(q.max(),q.min())   
      
#     # dataset = h5_Dataset('Z:\dataset\landslide\TrainData\img\\', 'Z:\dataset\landslide\TrainData\mask\\')
#     # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2)
#     # for x,y in train_loader:
#     #     print(x.shape), print(y.shape)
#     #     break
#     # print(torch.unique(y)) 
    
    # dataset = png_Dataset('X:\\SAMadapter\\image\\', 'X:\\SAMadapter\\label\\')
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True)

    # for i,(x,y) in enumerate(train_loader):
    #     print(x.shape), print(y.shape)
    #     # plot_func([x, y], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    #     if i==20:
    #         break
    # print(torch.unique(y))     