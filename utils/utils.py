#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:04:09 2018

@author: ali
"""
import torch
import os.path as path
import os
def squeeze_segm_map(segm_map,nclasses,batch_size,oh,ow):

    segm_map = segm_map.view(segm_map.size(0), int(segm_map.size(1)/nclasses),
    nclasses, segm_map.size(2), segm_map.size(3)).double()
    segm_map_ = torch.Tensor(batch_size, segm_map.size(1), oh, ow)
    
    _,segm_map_ = torch.max(segm_map.data,2)

    #segm_map_ = segm_map_.view(batch_size,segm_map.size(1), oh, ow)    
 
    return segm_map_ #output shape : (4,1,64,64)


def resize_batch(sInp):
    iszs = sInp.size()
    return torch.reshape(sInp, iszs[0], iszs[1]*iszs[2], iszs[3], iszs[4])



def get_all_batches(source_dir, sett):
    batch_list = []
    filedir = path.join(source_dir, sett)
    for file in os.listdir(filedir):
        if (file.find('batch_')!=-1):
            batch_list.append(file)

    return batch_list
#output is a list, containing all train or val .t7 files



def getBatch(batch):
    sample = torch.load(path.join('Data', 'val', batch))
    RGBs = sample['RGBs']
    frames = RGBs[:,:].cuda().mul(2/255).add(-1)
    #remember to double check this functions are in place to or, 
    #probably not and we have to add _ at the end of them
    segmE = sample['R8s'][:,:].cuda()
    return frames, segmE



def colorize(inp, colormap):
    colorized = torch.zeros(3,inp.size[1],inp.size[2])
    for ii in range(inp.size[1]):
        for jj in range (inp.size[2]):
            colorized[:,ii,jj] = torch.Tensor(colormap[inp[0][ii][jj]])
    return colorized


def display_segm(segm,  nb, filename, img, colormap, framed):
    dm = len(segm.size())
    ob,of,oh,ow = segm.size[0],segm.size[1],segm.size[dm-2],segm.size[dm-1]
    for n in range (of):
        for b in range(ob):
            colored = colorize(segm[b][n].double(), colormap)
            new_colored = torch.Tensor(3, oh, ow).fill(0)
            if (framed):
                new_colored[0].fill(1)
                new_colored[:,2:oh-2,2:ow-2]=colored[:,2:oh-2,2:ow-2]
            else:
                new_colored = colored.clone()
            
            if (img != None):
                imgcopy = img.clone()
                saved = new_colored.add(imgcopy[b][n].add(1).div(2).double())
                image.save(filename+'_'+str(n+nb)+'.png', saved )
            else:
                image.save(filename+'_'+str(n+nb)+'.png', new_colored)


def display_imgs(todisp, img, nb, filename):
    for k in range(img.size[1]):
        for b in range(img.size[0]):
            todisp[len(todisp)] = img[b][k][0:3]
            if (save):
                image.save(filename+'_'+str(k+nb)+'.png',img[b][k][0:3])



















