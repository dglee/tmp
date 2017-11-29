#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:42:42 2017

@author: dglee
"""
#=====================================================================================
# Global Parameters
#=====================================================================================
#Video Data
tr_video_path = '/home/dglee/Datasets/MSR_VTT_dataset/TrainValVideo'
ts_video_path = '/home/dglee/Datasets/MSR_VTT_dataset/TestVideo'

#Attributes: objects
tr_att_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/TrainVal_objects'
ts_att_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/Test_objects'

#Visual Feature: Inception last
#tr_visf_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/TrainVal_Inception'
#ts_visf_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/Test_Inception'

#Visual Feature: VGG fc6
tr_visf_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/TrainVal_VGG'
ts_visf_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/Test_VGG'

#Motion Feature: C3D FC6 sports1m_finetuning_ucf101 16frame, 8 overllapped
tr_c3d_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/TrainVal_C3D'
ts_c3d_path = '/home/dglee/Datasets/MSR_VTT_dataset/Features/Test_C3D'

#Ground truth captions
tr_gt_path='/home/dglee/Datasets/MSR_VTT_dataset/train_annotation/train_val_videodatainfo.json'
ts_gt_path='/home/dglee/Datasets/MSR_VTT_dataset/test_annotation/test_videodatainfo.json'

#Model saving path
model_path = '/home/dglee/Cap_Experiments/MSR_VTT/models'
#model_path = './models'
loss_img_path = '/home/dglee/Cap_Experiments/MSR_VTT/loss_img'


#=======================================================================================
# Train Parameters
#=======================================================================================
#VGG
dim_image = 4096

#Inception-Resnet
#dim_image = 1536
dim_hidden= 500

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80
n_attribute_category = 90

length_c3d = int((n_frame_step / 16) - 1)
dim_c3d = 4096
c3d_space = 16

save_every_n_epoch = 0.05
n_epochs = 10
batch_size = 50
learning_rate = 0.0001
#=======================================================================================
ts_batch_size = 10


