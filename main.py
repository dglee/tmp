# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf

import pandas as pd
import numpy as np
import os
import ipdb
import time

import cv2
import pickle

from model import Video_Caption_Generator
from scipy.stats import threshold
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

#from keras.preprocessing import sequence

    
#Read corpus csv file
def get_video_data(corpus_data_path, attribute_data_path, train_ratio = 0.9):
    video_data = pd.read_csv(corpus_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)    
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(attribute_data_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    
    unique_filename = video_data['video_path'].unique()
    train_len = int(len(unique_filename)*train_ratio)
    
    train_vids = unique_filename[:train_len]    
    test_vids = unique_filename[train_len:]
    
    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]
    
    return train_data, test_data
    

def preProBuildWordVocab(sentence_iterator, word_count_threshold=10): # borrowed this function from NeuralTalk
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
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector




###### class name of detected object ######
#category_index = np.load('./data/detection_category_Index.npy')
def load_attribute():    
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('avi'), videos)
    
   
    s_att_lst = []
    t_att_lst = []
    for video in videos:
        print video
            
        attribute_fullpath = os.path.join(attr_save_path, video + '.npz')
        att_feat = np.load(attribute_fullpath)
        
        #box_list (1,100,4)
        
        
        #box_list = att_feat['arr_0'].squeeze(1) #(frame_length, 100, 4)
        score_list = att_feat['arr_1'].squeeze(1) #(frame_length, 100)
        class_list = att_feat['arr_2'].squeeze(1) #(frame_length, 100)
        #ndeteciton_list = att_feat['arr_3']
        
        feat_fullpath = os.path.join(vis_feat_save_path, video + '.npy')
        #vis_feat = np.load(feat_fullpath)
        
        frame_length = np.size(score_list,0)    
        #if frame_length > 80:
        frame_indices = np.linspace(0, frame_length, num=n_frame_step, endpoint=False).astype(int)
            #Length normalization and top19 result
            #score_list = score_list[frame_indices,:18]
            #class_list = class_list[frame_indices,:18]
        score_list = score_list[frame_indices]
        class_list = class_list[frame_indices]
        
        #theresholding detection result
        t_score = threshold(score_list, 0.3)
        
        ################################################################
        #Generate spatiail attention based on confidence score
        # 90 = object category number
        s_att = np.zeros([n_frame_step, n_attribute_category])
        #s_att_count = np.zeros([n_frame_step, n_attribute_category])
        for i in range(n_frame_step):
            for j in range(100):
                if t_score[i,j] == 0:
                    continue
                
                #array index start from 0
                s_att[i, int(class_list[i,j])-1] += t_score[i, j]
                #s_att_count[i, int(class_list[i,j])-1] += 1
        
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
        
    # Output 
    # norm_s_att: 80 x 90
    # norm_t_att: 80
    
    return s_att_lst, t_att_lst


#=====================================================================================
# Global Parameters
#=====================================================================================
video_path = '/media/ssd1tb/YouTubeClips_encoding_cmb'
attr_save_path = '/media/ssd1tb/YouTube_Cap'
vis_feat_save_path = '/media/ssd1tb/YouTubeClips_features_80frame'
corpus_data_path='./data/video_corpus.csv'
model_path = './models'
#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 1000

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80
n_attribute_category = 90

n_epochs = 1000
batch_size = 50
learning_rate = 0.0001
#=======================================================================================

def caption_parsing(captions):
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)    
    return captions

def trigger_to_captions(current_captions, wordtoix):
    for idx, each_cap in enumerate(current_captions):
        word = each_cap.lower().split(' ')
        if len(word) < n_caption_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'
        else:
            new_word = ''
            for i in range(n_caption_lstm_step-1):
                new_word = new_word + word[i] + ' '
            current_captions[idx] = new_word + '<eos>'

    current_caption_ind = []
    for cap in current_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                current_word_ind.append(wordtoix[word])
            else:
                current_word_ind.append(wordtoix['<unk>'])
        current_caption_ind.append(current_word_ind)

    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
    current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
    current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
    nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )
    
    return current_captions, current_caption_matrix, current_caption_masks, nonzeros
    

def load_voca():
    #=======================================================================================
    #Load data, preprocessing on video corpus
    #=======================================================================================
    train_data, _ = get_video_data(corpus_data_path, vis_feat_save_path)
    captions = train_data['Description'].values
    captions = caption_parsing(captions)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)
    np.save("./data/wordtoix", wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save("./data/bias_init_vector", bias_init_vector)
    #=======================================================================================
    #wordtoix2 = np.load("./data/wordtoix.npy")
    #ixtoword = np.load('./data/ixtoword.npy')
    #bias_init_vector = np.load("./data/bias_init_vector.npy")
    
    return train_data, captions, wordtoix, ixtoword, bias_init_vector

def load_attention():
    ###temporal attention and spatial attention
    s_att, t_att = load_attribute()
    np.save("./data/s_attention", s_att)
    np.save("./data/t_attention", t_att)
    #=======================================================================================
    return s_att, t_att


def train():
    #=======================================================================================
    train_data, captions, wordtoix, ixtoword, bias_init_vector, = load_voca()
    
    #=======================================================================================
    #s_att, t_att = load_attention()
    s_att = np.load("./data/s_attention.npy")
    t_att = np.load("./data/t_attention.npy")
    #=======================================================================================
    
    print "Generate Model\n"
    model = Video_Caption_Generator(
            dim_image = dim_image, #attribute set
            n_words = len(wordtoix),
            dim_hidden = dim_hidden,
            batch_size = batch_size,
            n_lstm_step = n_frame_step,
            n_video_lstm_step = n_video_lstm_step,
            n_caption_lstm_step = n_caption_lstm_step,
            n_attribute_category = n_attribute_category,
            bias_init_vector = bias_init_vector)    
    print "Generate Model Done\n"
    
    print "Build Model\n"
    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_spatial_att, tf_temporal_att, tf_probs =  model.build_model()    
    print "Build Model Done\n"
    
    sess = tf.InteractiveSession()
    
    saver = tf.train.Saver(max_to_keep = 100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()
    
    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []
    
    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []
        
        index = list(train_data.index)
        #np.random.shuffle(index)
        train_data = train_data.ix[index]
        
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
        current_train_data = current_train_data.reset_index(drop=True)
        e_start_time = time.time()
        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):
            start_time = time.time()
            
            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values
            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)
            current_video_masks = np.zeros((batch_size, n_video_lstm_step))
            
            current_s_att = s_att[start:end]
            current_t_att = t_att[start:end]
            
            for ind, feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1
    
            current_captions = current_batch['Description'].values
            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            
            current_captions = caption_parsing(captions = current_captions)
            
            current_captions, current_caption_matrix, current_caption_masks, nonzeros = trigger_to_captions(current_captions, wordtoix)
            
            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
            
            fordebug = open('fordebug.txt', 'w')
            fordebug.write(str(current_s_att.shape) + "&" + str(current_t_att.shape) + "&" + str(current_caption_matrix.shape) + "&" + str(index) + '\n')
            fordebug.write(str(current_videos))
            fordebug.close()
            
            probs_val = sess.run(tf_probs, feed_dict={
                    tf_video:current_feats,
                    tf_caption: current_caption_matrix,
                    tf_spatial_att: current_s_att,
                    tf_temporal_att: current_t_att
                    })
            
            _, loss_val = sess.run([train_op, tf_loss], feed_dict={
                    tf_video: current_feats,
                    tf_video_mask: current_video_masks,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks,
                    tf_spatial_att: current_s_att,
                    tf_temporal_att: current_t_att
                    })
             #spatial_att = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step, self.n_attribute_category]) 
             #temporal_att = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step]) 
            loss_to_draw_epoch.append(loss_val)
            print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
            
        
        print ' Epoch time: ', str((time.time() - e_start_time))
        
        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_dir = "./loss_imgs"
        plt_save_img_name = str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 10) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    loss_fd.close()
            

if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    