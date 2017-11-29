# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
import json

from pandas.io.json import json_normalize
from model import Video_Caption_Generator
from parameters import *

coco_caption_dir = "./coco_eval/"
sys.path.insert(0, coco_caption_dir)
from cocoeval import COCOScorer, suppress_stdout_stderr

    
#Read corpus csv file
def get_video_data(tr_gt_path, ts_gt_path, tr_visf_path, ts_visf_path, tr_c3d_path, ts_c3d_path):
    
    #Training data
    with open(tr_gt_path) as data_file:
        data = json.load(data_file)
    
    sentences = json_normalize(data['sentences'])
    videos = json_normalize(data['videos'])

    train_vids = sentences.loc[sentences["video_id"].isin(videos[videos['split'] == "train"]["video_id"])]
    val_vids = sentences.loc[sentences["video_id"].isin(videos[videos['split'] == "validate"]["video_id"])]
    
    #visual feature
    train_vids['vis_feat_path'] = train_vids['video_id'].map(lambda x: os.path.join(tr_visf_path, x + ".mp4.npy"))  
    val_vids['vis_feat_path'] = val_vids['video_id'].map(lambda x: os.path.join(tr_visf_path, x + ".mp4.npy"))  
    
    #c3d feature
    train_vids['c3d_feat_path'] = train_vids['video_id'].map(lambda x: os.path.join(tr_c3d_path, x + ".mp4.npy"))  
    val_vids['c3d_feat_path'] = val_vids['video_id'].map(lambda x: os.path.join(tr_c3d_path, x + ".mp4.npy"))  

    #Test data
    with open(ts_gt_path) as data_file:
        data = json.load(data_file)
        
    sentences = json_normalize(data['sentences'])
    videos = json_normalize(data['videos'])    
    test_vids = sentences.loc[sentences["video_id"].isin(videos[videos['split'] == "test"]["video_id"])]
    test_vids['vis_feat_path'] = test_vids['video_id'].map(lambda x: os.path.join(ts_visf_path, x + ".mp4.npy"))  
    test_vids['c3d_feat_path'] = test_vids['video_id'].map(lambda x: os.path.join(ts_c3d_path, x + ".mp4.npy"))  

    return train_vids, val_vids, test_vids

    

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    np.save("./data/wordtoix", wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save("./data/bias_init_vector", bias_init_vector)
    

    return wordtoix, ixtoword, bias_init_vector



###### class name of detected object ######
#category_index = np.load('./data/detection_category_Index.npy')
def load_attribute(video_path, attr_save_path, vis_feat_save_path, isTraining):    
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('mp4'), videos)
    
   
    s_att_lst = []
    t_att_lst = []
    vid_name_lst = []
    for video in videos:
        print video
            
        attribute_fullpath = os.path.join(attr_save_path, video + '.npz')
        att_feat = np.load(attribute_fullpath)
        
        
        score_list = att_feat['arr_1'].squeeze(1) #(frame_length, 100)
        class_list = att_feat['arr_2'].squeeze(1) #(frame_length, 100)

        feat_fullpath = os.path.join(vis_feat_save_path, video + '.npy')
        
        frame_length = np.size(score_list,0)    
        #if frame_length > 80:
        frame_indices = np.linspace(0, frame_length, num=n_frame_step, endpoint=False).astype(int)
        score_list = score_list[frame_indices]
        class_list = class_list[frame_indices]
        
        #theresholding detection result
        #t_score = threshold(score_list, 0.3, 100)
        t_score = np.clip(score_list, 0.3, 100)
        
        ################################################################
        #Generate spatiail attention based on confidence score
        # 90 = object category number
        s_att = np.zeros([n_frame_step, n_attribute_category])

        for i in range(n_frame_step):
            for j in range(100):
                if t_score[i,j] == 0:
                    continue
                
                #array index start from 0
                s_att[i, int(class_list[i,j])-1] += t_score[i, j]
        
        #Score normalization Necessary?
        row_sum = s_att.sum(axis=1)
        row_sum[row_sum == 0.0] = 0.01 #remove zero
        norm_s_att = s_att / row_sum[:, np.newaxis]
        ################################################################
        
        ################################################################
        #Temporal attention weight
        # Have to be changed
        # Its temporal code for implementation
        ################################################################
        col_sum = s_att.sum(axis=1)
        col_sum[col_sum == 0.0] = 0.01 #remove zero
        norm_t_att = col_sum / col_sum.sum()
        ################################################################
        
        s_att_lst.append(norm_s_att)
        t_att_lst.append(norm_t_att)
        vid_name_lst.append(feat_fullpath)
        
        
        
    if isTraining == True:
        np.save("./data/tr_s_attention", s_att_lst)
        np.save("./data/tr_t_attention", t_att_lst)
        np.save("./data/tr_att_name", vid_name_lst)
    else:
        np.save("./data/ts_s_attention", s_att_lst)
        np.save("./data/ts_t_attention", t_att_lst)
        np.save("./data/ts_att_name", vid_name_lst)
	
    # norm_s_att: 80 x 90
    # norm_t_att: 80

    return s_att_lst, t_att_lst, vid_name_lst

def caption_parsing(captions):
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    return captions
    

def get_att_batch(current_batch, b_size, s_att, t_att, att_name):
    
    #c_video_id = current_batch['vis_feat_path'].values
    c_video_id = current_batch
    
    current_s_att = np.zeros([b_size, n_video_lstm_step, n_attribute_category] )
    current_t_att = np.zeros([b_size, n_video_lstm_step])
    
    for i in range(b_size):
        idx = np.where(att_name == c_video_id[i])[0][0]
        current_s_att[i] = s_att[idx]
        current_t_att[i] = t_att[idx]
    
    return current_s_att, current_t_att


def draw_loss_graph(epoch, step_count, model_counter, loss_to_draw, loss_to_draw_val):
    plt_save_dir = loss_img_path
    plt_save_img_name = str(model_counter) + '.png'
    plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
    plt.plot(range(len(loss_to_draw_val)), loss_to_draw_val, color='b')
    plt.grid(True)
    plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))     


def print_remaining_time(epoch, start_time, step_summary, step_count, n_step, val_loss, train_loss):
    remain_epoch = n_epochs - epoch    
    step_time = time.time() - start_time
    step_summary[step_count] = step_time
    average_step_time = np.sum(step_summary)/(step_count+ 1)
    remain_time = (average_step_time * remain_epoch * n_step) + (average_step_time * (n_step-step_count))
    m, s = divmod(remain_time, 60) 
    h, m = divmod(m, 60) 
    
    print "=============================================================================================="
    print("Epoch: %d/%d Step: %d/%d Remain: %d h %d m  step: %.2f s  val_loss: %.3f  tr_loss: %.3f"%(epoch+1, n_epochs, step_count, n_step, h, m, step_time, val_loss, train_loss))
    print "==============================================================================================\n"
    
    return step_summary



def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    non_ascii_count = 0
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        try:
            row[0].encode('ascii', 'ignore').decode('ascii')
        except UnicodeDecodeError:
            non_ascii_count+=1
            continue
        if row[1] in gts:
            gts[row[1]].append({u'image_id': row[1], u'cap_id': len(gts[row[1]]), u'caption':row[0].encode('ascii', 'ignore').decode('ascii')})
        else:
            gts[row[1]] = []
            gts[row[1]].append({u'image_id': row[1], u'cap_id': len(gts[row[1]]), u'caption':row[0].encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print "=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20
    return gts


    
    
