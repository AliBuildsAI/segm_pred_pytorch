#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:55:32 2018

@author: ali
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
class Model(nn.Module):
    def __init__(self,n_channels_input, n_channels_output,mod_size=4):
        super(Model, self).__init__()
        self.conv1_scale = nn.Conv2d(n_channels_input, 32*mod_size,kernel_size=3, stride=1, padding=1)
        self.conv2_scale = nn.Conv2d(32*mod_size, 64*mod_size,kernel_size=3, stride=1, padding=1)
        self.conv3_scale = nn.Conv2d(64*mod_size, 32*mod_size,kernel_size=3, stride=1, padding=1)
        self.conv4_scale = nn.Conv2d(32*mod_size, n_channels_output,kernel_size=3, stride=1, padding=1)
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(n_channels_input+n_channels_output, 32*mod_size,kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32*mod_size, 64*mod_size,kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64*mod_size, 32*mod_size,kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32*mod_size, n_channels_output,kernel_size=5, stride=1, padding=2)
    def forward(self, frames_tensor):
        #frames_downsample = self.down_sample(frames_tensor)
        x = F.relu(self.conv1_scale(frames_tensor[0]))
        x = F.relu(self.conv2_scale(x))
        x = F.relu(self.conv3_scale(x))
        x_scale = self.conv4_scale(x)
        scale_pred_upsample=self.upsample(x_scale)
        large_scale_input=torch.cat((frames_tensor[1],scale_pred_upsample),1)
        x = F.relu(self.conv1(large_scale_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return [x_scale, x+scale_pred_upsample]


def get_pyr_preprocessor(dataset):

    avg_pool=[]
    scaleList=[2,1]
    output_list=[]
    for i in range(len(scaleList)):
        avg_pool.append(nn.AvgPool2d(scaleList[i], stride=scaleList[i]))
    for i in range(len(avg_pool)):
        output_list.append(avg_pool[i](dataset))
    return output_list


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class gdl1_l1_loss(_Loss):
    def __init__(self,size_average=False):
        super(gdl1_l1_loss, self).__init__(size_average)
        self.y_i_1=nn.ZeroPad2d((0,0,0,-1))
        self.y_i_2=nn.ZeroPad2d((0,0,-1,0))
        self.y_j_1=nn.ZeroPad2d((0,-1,0,0))
        self.y_j_2=nn.ZeroPad2d((-1,0,0,0))
        self.losss=0
        self.term1=[]
        self.term2=[]
        self.term3=[]
        self.term4=[]
        self.loss=nn.L1Loss()
    def forward(self, output, target):

        #self.losss=0
        #self.term1=[]
        #self.term2=[]
        #self.term3=[]
        #self.term4=[]

        #print(self.losss,len(self.term3))
        for i in range(2):
            _assert_no_grad(target[i])
            #print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            self.term1.append(torch.abs(self.y_i_1(output[i])-self.y_i_2(output[i])))
            self.term2.append(torch.abs(self.y_i_1(target[i])-self.y_i_2(target[i])))
        
            self.term3.append(torch.abs(self.y_j_1(output[i])-self.y_j_2(output[i])))
            self.term4.append(torch.abs(self.y_j_1(target[i])-self.y_j_2(target[i])))
           
            #print('fffffffffffffffffffffffffffffffffffffffffff')
            self.losss+=self.loss(self.term1[i],self.term2[i])+self.loss(self.term3[i],self.term4[i])+self.loss(output[i],target[i])
            #print('jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')

        return self.losss