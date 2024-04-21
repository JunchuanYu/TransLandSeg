import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
from torch.utils.data.sampler import SubsetRandomSampler
from Evaluator import evaluator
from losses import calc_loss
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from loader import png_Dataset
from result_to_csv import result_to_csv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training On Device: ", device)
local_rank = 0
parser = argparse.ArgumentParser()
parser.add_argument('--config', default="translandseg.yaml")
parser.add_argument('--resume', default='checkpoint', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', 
                    help='total epochs')
parser.add_argument('--data_path_img',
                    type=str, metavar='data', help='path to dataset')
parser.add_argument('--data_path_label',
                    type=str, metavar='data', help='path to dataset')
args = parser.parse_args(args=[])

with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def make_data_loader():
    dataset = png_Dataset(args.data_path_img, args.data_path_label)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.3 * num_train))+1

    if True:
        np.random.seed(42)
        torch.manual_seed(42)  
        np.random.shuffle(indices)    
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=train_sampler, shuffle=False,
                                            num_workers=2, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=valid_sampler,shuffle=False,
                                            num_workers=2, pin_memory=True) 
    return train_loader, valid_loader

def make_data_loaders():
    train_loader,val_loader = make_data_loader()
    return train_loader, val_loader

def prepare_training():
    model = models.make(config['model']).cuda()
    optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=config.get('lr_min'))

    return model, optimizer, lr_scheduler


train_loader, val_loader = make_data_loaders()
x_list=[]
y_list=[]
for  i,(x,y) in enumerate(train_loader) :

    if i == 5:
        break
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    x1 = (x* torch.tensor(std).view(1, 3, 1, 1)) + torch.tensor(mean).view(1, 3, 1, 1)
    x_list.append(x1)
    y_list.append(y)
x_list = np.concatenate(x_list, axis=0)
y_list = np.concatenate(y_list, axis=0)
x_arr=np.array(x_list)
y_arr=np.array(y_list)
print(np.max(x_arr), np.min(x_arr),np.unique(y_arr))
def plot_func(data_list, n=2, camp='gray', mean=None, std=None):

    for m in range(n):
        fig=plt.figure(figsize=(30,8))
        for i in range(10):
            plt.subplot(1,10,i+1)
            if 'cuda' in data_list[m].device.type:
                img = data_list[m][i,:,:,:].cpu().numpy().transpose(1, 2, 0)
            else:
                img = data_list[m][i,:,:,:].numpy().transpose(1, 2, 0)
            if camp == 'gray' and img.shape[-1] == 1:
                plt.imshow(img, cmap=camp)
            elif mean and std and img.shape[-1] == 3:
                plt.imshow(img *std + mean)
            else:
                plt.imshow(img)
        plt.show()
x_list = torch.tensor(x_list)
y_list = torch.tensor(y_list)
plot_func([x_list, y_list], n=2) 
model, optimizer, lr_scheduler = prepare_training()
model = model.cuda()
sam_checkpoint = torch.load(config['sam_checkpoint'])
model.load_state_dict(sam_checkpoint, strict=False)

for name, para in model.named_parameters():
    if "image_encoder" in name and "prompt_generator" not in name:
        para.requires_grad_(False)
def training(train_loader, model, optimizer, epoch, args):
    train_loss = 0.0
    total_num = 0.0
    model.train()
    evaluator.reset()
    tbar = tqdm(train_loader, desc='Training>>>>>>>')
    for i, (x,y) in enumerate(tbar):
        x.type(torch.cuda.FloatTensor)
        y.type(torch.cuda.FloatTensor)
        image, target = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model.infer(image)
        loss = calc_loss(output, target.float())
        loss.backward()
        optimizer.step()
        total_num +=  x.size(0)
        train_loss += loss.data.cpu().numpy() * x.size(0)
        pred = torch.sigmoid(output)  
        pred = torch.where(pred > 0.5, 1, 0) # 二值化
        evaluator.add_batch(target, pred)
        accuracy = evaluator.OverallAccuracy()
        presion = evaluator.Precision() # 一个列表包含每个类的查准率
        recall = evaluator.Recall() # 一个列表包含每个类的查全率
        F1score = evaluator.F1Score()  # 一个列表包含每个类的F1值
        Iou = evaluator.IntersectionOverUnion() # 一个列表包含每个类的Iou值
        FWIou = evaluator.Frequency_Weighted_Intersection_over_Union() 
        mIou = evaluator.MeanIntersectionOverUnion()
        tbar.set_description('Training  ->>>- Epoch: [%3d]/[%3d]  Train loss: %.4f  Train Accuracy:%.3f  ' % (
            epoch+1, args.epochs, train_loss / total_num, accuracy))
    
    result_dict = {'accuracy': accuracy, 'presion': presion, 'recall': recall, 'F1score': F1score, 'Iou': Iou, 'FWIou': FWIou, 'mIou': mIou, 'loss': train_loss/total_num}   
    return train_loss/total_num, accuracy, result_dict


def validation(val_loader, model, epoch, args):
    
    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader, desc='Validation>>>>>>>')
    test_loss = 0.0
    total_num = 0.0
    for i, (x,y) in enumerate(tbar):

        x.type(torch.cuda.FloatTensor)
        y.type(torch.cuda.FloatTensor)
        image, target = x.to(device), y.to(device)
        with torch.no_grad():
            output = model.infer(image)
        loss = calc_loss(output, target.float())
        test_loss += loss.data.cpu().numpy() * x.size(0)
        total_num +=  x.size(0)
        
        pred = torch.sigmoid(output)  
        pred = torch.where(pred > 0.5, 1, 0) 

        evaluator.add_batch(target, pred)
        accuracy = evaluator.OverallAccuracy()
        presion = evaluator.Precision() 
        recall = evaluator.Recall() 
        F1score = evaluator.F1Score()  
        Iou = evaluator.IntersectionOverUnion() 
        FWIou = evaluator.Frequency_Weighted_Intersection_over_Union() 
        mIou = evaluator.MeanIntersectionOverUnion()

        tbar.set_description('Validation->>>- Epoch: [%3d]/[%3d]  Valid loss: %.4f  Valid Accuracy:%.3f  ' % (
            epoch+1, args.epochs, test_loss / total_num, accuracy)) 
        
    result_dict = {'accuracy': accuracy, 'presion': presion, 'recall': recall, 'F1score': F1score, 'Iou': Iou, 'FWIou': FWIou, 'mIou': mIou, 'loss': test_loss/total_num}   
    return test_loss/total_num, accuracy, result_dict  


def save_checkpoint(state, is_best, dir=None, filename='checkpoint.pth.tar'):
    if dir:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print('File ['+ dir + '] Created successfully')
    torch.save(state, dir + filename)
    if is_best:
        shutil.copyfile(dir + filename, 'model_best.pth.tar')
        


def early_stopping(valid_loss, epoch, args, valid_loss_min=[np.inf], i_valid=[0]):
    if valid_loss <= valid_loss_min[-1]:
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, is_best=False,  dir = args.resume, filename='checkpoint_{}_{}.pth.tar'.format(epoch+1, args.epochs))

        if round(valid_loss, 4) == round(valid_loss_min[-1], 4):
            print(i_valid[0])
            i_valid[0] += 1
        valid_loss_min[-1] = valid_loss
        if i_valid[0] == 10:
            print('Early stopping')
            return True
# Define Evaluator
evaluator = evaluator(2)

train_losses,train_accuaryes, test_losses, test_accuracyes, train_results, test_reselts= [], [], [], [], [], []
for epoch in range(args.epochs):
    
    since = time.time()
    
    # train for one epoch
    tanin_loss, train_accuary, train_result = training(train_loader, model, optimizer, epoch, args)
    test_loss, test_accuracy, test_reselt = validation(val_loader, model, epoch, args)
    
    # cosine learning rate scheduler
    lr_scheduler.step()
    # log to lists
    train_losses.append(tanin_loss)
    train_accuaryes.append(train_accuary)
    test_losses.append(test_loss)
    test_accuracyes.append(test_accuracy)
    train_results.append(train_result)
    test_reselts.append(test_reselt)
    # early stopping and save model
    if early_stopping(test_loss, epoch, args):
        break

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
### 可视化训练过程的loss和accuracy
def plot_fig(train_losses, valid_losses, train_accuracyes, valid_accuracyes, outdir):
    # 创建文件
    if not os.path.exists(os.path.dirname(outdir)):
        os.makedirs(os.path.dirname(outdir))
        # print('File ['+ os.path.split(outdir)[0] + '] Created successfully')
    plt.style.use("ggplot")
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, 1 + args.epochs), np.array(train_losses), label="train_loss")
    plt.plot(np.arange(1, 1 + args.epochs), np.array(valid_losses), label="val_loss")
    plt.plot(np.arange(1, 1 + args.epochs), np.array(train_accuracyes), label="train_accuracyes")
    plt.plot(np.arange(1, 1 + args.epochs), np.array(valid_accuracyes), label="valid_accuracyes")
    plt.ylim(0, 1)
    plt.title("Training and Validation Loss / Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(outdir + '_[Loss-Accuracy]_epoch.png')

plot_fig(train_losses, test_losses, train_accuaryes, test_accuracyes, args.resume + "picture/SAM_prompt2z_" + str(epoch))
result_to_csv(train_results, out_file=args.resume + 'picture/SAM_prompt2z_train_'+ str(args.epochs) + '.csv')
result_to_csv(test_reselts, out_file=args.resume + 'picture/SAM_prompt2z_valid_'+ str(args.epochs) + '.csv')

