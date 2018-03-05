#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 02:08:35 2018

@author: ali
"""

import torch
torch.backends.cudnn.benchmark=True

import torchnet as tnt
from torch.utils.serialization import load_lua
import sys
#remember to import GPU related stuff
#remember to install cudnn in google cloud  (DONE)
import time

import argparse
import datetime
import os.path as path
import os
import random
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from utils.dataset import *
from utils.utils import *
from utils.model import *
from utils.metrics import *
from tqdm import trange
from torchnet.meter import ConfusionMeter
#REMEMBER TO LOG EACH PRINT LINE IN A FILE

now = datetime.datetime.now()
time_now=now.strftime("%d%B%Y_%H_%M_%S")

parser = argparse.ArgumentParser(description='PyTorch implementation of SegmPred')
parser.add_argument('--devid', default=[0,1,2,3], help='GPU id')#To be changed
parser.add_argument('--save-dir',default='saves',help='Directory to save the data')
parser.add_argument('--data-dir',default='./Data/',help='dataset path')
parser.add_argument('--optim', default='sgd', help='Optim scheme')
parser.add_argument('--n-epoches', default=5000, help='Number of epoches')
parser.add_argument('--n-iters', default=1000, help='Number of training iterations per epoch')#WHAT IS THIS?????
parser.add_argument('--n-iters-test', default=25, help='Number of testing iterations per epoch')
parser.add_argument('--lr', default=0.01, help= 'Learning rate of the frame generator')
parser.add_argument('--batch-size', default=4, help= 'Minibatch size')
parser.add_argument('--n-input-frames', default=4, help='Number of input frames (excluding prediction)')
parser.add_argument('--n-target-frames', default=1, help='Number of predicted frames') #This one and the one above should be changed for long-term and mid-term prediction
parser.add_argument('--h-input', default=64, help='Frame height')
parser.add_argument('--w-input', default=64, help='Frame width')
#parser.add_argument('--crit', default='gdll1', help='loss : Abs, MSE, GDL, gdll1, SpatialClassNLL')
parser.add_argument('--save-freq', default=40, help='saving after this number of iterations')
parser.add_argument('--no-cuda', default=False,
                    help='disables CUDA training')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()
print('Running with training options:', opt)

with open(time_now+'.txt', 'a') as f:
    f.write('Running with training options:\n{}'.format(opt))


opt.nscales = 2 
opt.segm = 1 #WHAT'S THIS??
#Answer: The archetucture depends on this (Probably won't be useful in my code!)
torch.set_num_threads(1)
torch.manual_seed(1)
if opt.cuda:
    torch.cuda.manual_seed(1)
torch.cuda.device(opt.devid)


if (path.exists(opt.save_dir)):
    os.system('rm -r ' + opt.save_dir + '.bkp')
    os.system('mv ' + opt.save_dir + ' ' + opt.save_dir + '.bkp')
    print('Copied existing '+opt.save_dir+' into '+opt.save_dir+'.bkp')
    with open(time_now+'.txt', 'a') as f:
        f.write('Copied existing '+opt.save_dir+' into '+opt.save_dir+'.bkp')
os.system('mkdir -p ' + opt.save_dir)

opt.n_channels = nclasses #will be imported from utils.dataset
opt.n_classes = nclasses

train_batch_list = get_all_batches(opt.data_dir,'train') #IN utils.utils
#train_batch_list is a list of length 99, containing all .t7 training files
val_batch_list = get_all_batches(opt.data_dir,'val') 
#val_batch_list is a list of length 125, containing all .t7 validation files
#val_batch_list was a table (table in lua = dict in python)


if (opt.n_iters_test>len(val_batch_list)):#Won't happen in our case
    print('Only '+str(len(val_batch_list))+ ' test batches available')

    with open(time_now+'.txt', 'a') as f:
        f.write('Only '+str(len(val_batch_list))+ ' test batches available')


if (opt.n_iters_test==0):#Won't happen in our case
    opt.n_iters_test=len(val_batch_list)
print('Training on '+str(len(train_batch_list))+ ' batches')
print('Validation on '+str(len(val_batch_list))+ ' batches')

with open(time_now+'.txt', 'a') as f:
    f.write('Training on '+str(len(train_batch_list))+ ' batches')
    f.write('Validation on '+str(len(val_batch_list))+ ' batches')



#creating the model 
model = Model(n_channels_input=opt.n_classes*opt.n_input_frames, n_channels_output=opt.n_classes,mod_size=4)
model = torch.nn.DataParallel(model).cuda()

#defining the loss
#loss=gdl1_l1_loss()
#loss.cuda()
#loss=nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum=0.9)
#shortcuts
ob = opt.batch_size
tf = opt.n_target_frames
inpf = opt.n_input_frames
hi, wi = opt.h_input, opt.w_input
ch = opt.n_channels
#confusion = tnt.SemSegmMeter{classes = classes}
#confusion=ConfusionMeter(len(classes))
confusion= sem_segm_meter(classes)

#to be implemented in utils.metrics  (DONE)

def get_batch(sett, iIter):
    if (sett == 'train'):
        iIter = random.randint(1, len(train_batch_list))
    else:
        iIter+=1
    #print(iIter,'in get batch')
    sample = load_lua(path.join(opt.data_dir, sett, 'batch_'+str(iIter)+'.t7'))
    if (sett == 'train'):
        if opt.cuda:
            segmInputE = sample['R8s'][:,0:inpf].cuda()
            #torch.Size([4(batch_size), 4(prevFrames), 19(nClasses), 64, 64])
            segmTargetE = sample['R8s'][:,inpf:inpf+tf].cuda()
            
        else:
            
            segmInputE = sample['R8s'][:,0:inpf]#.cuda()
        #torch.Size([4(batch_size), 4(prevFrames), 19(nClasses), 64, 64])
            segmTargetE = sample['R8s'][:,inpf:inpf+tf]#.cuda()
        #torch.Size([4(batch_size), 1(targetFrame), 19(nClasses), 64, 64])

    else:
        RGBs = sample['RGBs']
        #sample["RGBs"].shape: torch.Size([4 (batch_size), 5 (frames), 3(channels), 64(width), 64(height)])
        h = random.randint(0, oh-hi-1)#what are oh and ow?
        w = random.randint(0, ow-wi-1)
        
        if opt.cuda:
            segmInputE = sample['R8s'][:,0:inpf,:,h:h+hi-1,w:w+wi-1].cuda()
            segmTargetE = sample['R8s'][:,inpf:inpf+tf,:,h:h+hi-1,w:w+wi-1].cuda()
            
            
        else:
            segmInputE = sample['R8s'][:,0:inpf,:,h:h+hi-1,w:w+wi-1]#.cuda()
            segmTargetE = sample['R8s'][:,inpf:inpf+tf,:,h:h+hi-1,w:w+wi-1]#.cuda()
    segmTargetE.resize_(ob, tf*ch, wi, hi)
    segmInputE.resize_(ob, inpf*ch, wi, hi)
    segmInputE, segmTargetE = Variable(segmInputE,requires_grad=True), Variable(segmTargetE,requires_grad=False)
    
    #segmTargetE.
    return get_pyr_preprocessor(segmInputE), get_pyr_preprocessor(segmTargetE)
def training(iIter):
    
    model.train()
    inputt, target = get_batch('train', iIter)
    optimizer.zero_grad()
    output = model.forward(inputt)
    loss= gdl1_l1_loss()
    l2err = loss.forward(output, target)
    #print(l2err)
    l2err.backward()
    optimizer.step()     
    return l2err.data[0]
            

def testing():
    confusion.reset()
    model.eval()
    for j in range(99):#remember to change this to opt.n_iters_test
        #the first 25 val batch are used for validation
        inputt, target = get_batch('train', j)#remember to change it to 'val', instead of 'train'
        #inputt, target = Variable(inputt), Variable(target)
        pred = model.forward(inputt)
        spredF = squeeze_segm_map(pred[-1].clone(),opt.n_classes,ob,hi,wi)
        stargetF = squeeze_segm_map(target[-1].clone(),opt.n_classes,ob,hi,wi)
        confusion.add(spredF.view(-1), stargetF.view(-1))
    return confusion.avgIOU()




def save_checkpoint(state, filename):
    torch.save(state, filename)

#main training loop 
for iEpoch in range(opt.n_epoches):#5000
    sumGenErr = 0
    start = time.time()
    for iIter in range(opt.n_iters):#1000
        
        sumGenErr = sumGenErr + training(iIter)

    avgGenErr = sumGenErr / opt.n_iters
    print("Epoch ",iEpoch,'/',opt.n_epoches,"; Generator error = ",avgGenErr)
    with open(time_now+'.txt', 'a') as f:
        f.write("Epoch "+str(iEpoch)+'/'+str(opt.n_epoches)+"; Generator error = "+str(avgGenErr))
    if (iEpoch % opt.save_freq == 0):
        print('Saving the model...')
        with open(time_now+'.txt', 'a') as f:
            f.write('Saving the model...')
            f.write(path.join(opt.save_dir,'model_'+str(iEpoch)+'_epochs.pth'))

        save_checkpoint({'epoch': iEpoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        },filename=path.join(opt.save_dir,'model_'+str(iEpoch)+'_epochs.pth'))
        
        
    print ("it took", time.time() - start, "seconds. (1000 iterations of training)")
#https://github.com/pytorch/examples/blob/0984955bb8525452d1c0e4d14499756eae76755b/imagenet/main.py#L139-L145
    start = time.time()
    IOU_val=testing()
    print ("it took", time.time() - start, "seconds. (testing)")
    
    print('Validation [IoU] ',IOU_val)
    with open(time_now+'.txt', 'a') as f:
        f.write('Validation [IoU] '+str(IOU_val))

