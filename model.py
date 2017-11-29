#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:54:39 2017

@author: dglee
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb

import cv2

import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn


from keras.preprocessing import sequence
#from ops import fc

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, dim_c3d, length_c3d, c3d_space, batch_size, n_lstm_step, n_video_lstm_step, n_caption_lstm_step, n_attribute_category, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.dim_c3d = dim_c3d
        self.length_c3d = length_c3d        
        self.n_lstm_step = n_lstm_step
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step
        self.n_attribute_category = n_attribute_category
        self.c3d_space = c3d_space
        
        
        #self.lstm_seq1 = tf.contrib.rnn.GRUCell(dim_hidden)
        #self.lstm_cap = tf.contrib.rnn.GRUCell(dim_hidden)

        self.lstm_seq1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')        
        
        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        
        self.encode_c3d_W = tf.Variable( tf.random_uniform([dim_c3d, dim_hidden], -0.1, 0.1), name='encode_c3d_W')
        self.encode_c3d_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_c3d_b')
        
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
        

    
    #Generate fully connected layer
    def fc(self, input, output_shape, activation_fn=tf.nn.relu, name="fc"):
        output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
        return output            
    
    #build_model for training
   
    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step, self.dim_image])  
        c3d_feat = tf.placeholder(tf.float32, [self.batch_size, self.length_c3d, self.dim_c3d]) 
        
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step])
        
        video_flat = tf.reshape(video, [-1, self.dim_image])
        c3d_flat = tf.reshape(c3d_feat, [-1, self.dim_c3d])
        
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b) # batch_size * n_lstm_step, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_step, self.dim_hidden])
        
        c3d_emb = tf.nn.xw_plus_b(c3d_flat, self.encode_c3d_W, self.encode_c3d_b) # batch_size * n_lstm_step, dim_hidden)
        c3d_emb = tf.reshape(c3d_emb, [self.batch_size, self.length_c3d, self.dim_hidden])
        
        state1 = tf.zeros([self.batch_size, self.lstm_seq1.state_size])
        state3 = tf.zeros([self.batch_size, self.lstm_cap.state_size])        
        
        
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        ##attribute information
        spatial_att = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step, self.n_attribute_category]) 
        temporal_att = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step]) 

        probs = []
        loss = 0.0
        
        vis_feat = tf.concat([image_emb, spatial_att], axis=2)
        
#        for i in range(self.length_c3d):
#            for j in range(10):
#                reuse_flag = False
#                if i > 0:
#                    tf.get_variable_scope()
#                    reuse_flag = True
#                with tf.variable_scope("LSTM_mid",reuse=reuse_flag):
#                    outputmid, state_mid = self.lstm_mid(vis_feat)
            
        
        
        idx = range(self.c3d_space -1, self.n_lstm_step-1, self.c3d_space)
        tmp_img_emb = []
        for i in range(self.length_c3d):
            tmp_img_emb.append(vis_feat[:,idx[2],:])
        
        img_input = tf.reshape(tmp_img_emb,[self.batch_size, self.length_c3d, self.dim_hidden + self.n_attribute_category])
        
        reshape_input = tf.concat([img_input, c3d_emb], axis=2)
         
#        f_c3d=[]
#        for i in range(self.n_lstm_step):
#            c3d_idx = i/8
#            f_c3d[:,i,:] = c3d_emb[:,c3d_idx,:]
        
        
        #reshape_input = tf.reshape(tf.concat([image_emb, spatial_att], axis=2),
        #                           [self.batch_size * self.n_lstm_step, self.dim_hidden + self.n_attribute_category])
        
        #G theta
        g_1 = self.fc(reshape_input, 500, name='g_1')
#        g_2 = self.fc(g_1, 256, name='g_2')
#        g_3 = self.fc(g_2, 256, name='g_3')
#        g_4 = self.fc(g_3, 256, name='g_4')        
        ############################################################
        #############elementwise sum is not implemented#############
        ############################################################
#       g_out = g_1
        #F_phi
        fc_1 = self.fc(g_1, 256, name='fc_1')
        fc_2 = self.fc(fc_1, 256, name='fc_2')
        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=True)
        fc_3 = self.fc(fc_2, self.dim_hidden, activation_fn=None, name='fc_3')
        #f_out = tf.reshape(fc_3, [self.batch_size, self.n_lstm_step, self.dim_hidden])
        f_out = tf.reshape(fc_3, [self.batch_size, self.length_c3d, self.dim_hidden])
                

        ##### Encoding #####
        #for i in range(self.n_video_lstm_step):
        for i in range(self.length_c3d):
            reuse_flag = False
            if i > 0:
                tf.get_variable_scope()
                reuse_flag = True
            with tf.variable_scope("LSTM_SEQ1",reuse=reuse_flag):
                output1, state1 = self.lstm_seq1(f_out[:,i,:], state1)
                #t_out = output1 * temporal_att[:,i]
                tmp_out = tf.transpose(output1, [1, 0])
                t_out = tf.multiply(tmp_out,temporal_att[:,i])
                t_out = tf.transpose(t_out, [1, 0])                
            with tf.variable_scope("LSTM_CAP",reuse=reuse_flag):    
                output3, state3 = self.lstm_cap(tf.concat([padding, t_out], axis=1), state3)
                
        ##### Decoding #####
        for i in range(self.n_caption_lstm_step):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i-1])            
            
            tf.get_variable_scope()
            with tf.variable_scope("LSTM_SEQ1", reuse=True):
                output1, state1 = self.lstm_seq1(padding, state1)

            with tf.variable_scope("LSTM_CAP", reuse=True):    
                output3, state3 = self.lstm_cap(tf.concat([current_embed, output1], axis=1), state3)

                
            labels = tf.expand_dims(caption[:,i], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], axis=1)
            outshape = tf.stack([self.batch_size, self.n_words])
            onehot_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output3, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy) / self.batch_size
            loss += current_loss

        return loss, video, c3d_feat, caption, caption_mask, spatial_att, temporal_att, probs

    #build_generator for test
    #def build_generator(self):
        
    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        c3d_feat = tf.placeholder(tf.float32, [1, self.length_c3d, self.dim_c3d]) 

        video_flat = tf.reshape(video, [-1, self.dim_image])
        c3d_flat = tf.reshape(c3d_feat, [-1, self.dim_c3d])
        
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        c3d_emb = tf.nn.xw_plus_b(c3d_flat, self.encode_c3d_W, self.encode_c3d_b) # batch_size * n_lstm_step, dim_hidden)
        c3d_emb = tf.reshape(c3d_emb, [1, self.length_c3d, self.dim_hidden])


        state1 = tf.zeros([1, self.lstm_seq1.state_size])
        state3 = tf.zeros([1, self.lstm_cap.state_size])        
        padding = tf.zeros([1, self.dim_hidden])


        ##attribute information
        spatial_att = tf.placeholder(tf.float32, [1, self.n_lstm_step, self.n_attribute_category]) 
        temporal_att = tf.placeholder(tf.float32, [1, self.n_lstm_step]) 
        
        generated_words = []
        probs = []
        embeds = []

        #reshape_input = tf.reshape(tf.concat([image_emb, spatial_att], axis=2),
        #                           [self.n_lstm_step, self.dim_hidden + self.n_attribute_category])
        vis_feat = tf.concat([image_emb, spatial_att], axis=2)
        
        idx = range(self.c3d_space -1, self.n_lstm_step-1, self.c3d_space)
        tmp_img_emb = []
        for i in range(self.length_c3d):
            tmp_img_emb.append(vis_feat[:,idx[2],:])
        
        img_input = tf.reshape(tmp_img_emb,[1, self.length_c3d, self.dim_hidden + self.n_attribute_category])
        
        reshape_input = tf.concat([img_input, c3d_emb], axis=2)        
          
        #G theta
        g_1 = self.fc(reshape_input, 500, name='g_1')
#        g_2 = self.fc(g_1, 256, name='g_2')
#        g_3 = self.fc(g_2, 256, name='g_3')
#        g_4 = self.fc(g_3, 256, name='g_4')        
        #F_phi
        fc_1 = self.fc(g_1, 256, name='fc_1')
        fc_2 = self.fc(fc_1, 256, name='fc_2')
        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=False)
        fc_3 = self.fc(fc_2, self.dim_hidden, activation_fn=None, name='fc_3')
        f_out = tf.reshape(fc_3, [1, self.length_c3d, self.dim_hidden])


        #for i in range(self.n_video_lstm_step):
        for i in range(self.length_c3d):
            reuse_flag = False
            if i > 0:
                tf.get_variable_scope()
                reuse_flag = True
            
            with tf.variable_scope("LSTM_SEQ1",reuse=reuse_flag):
                output1, state1 = self.lstm_seq1(f_out[:,i,:], state1)
                #t_out = output1 * temporal_att[:,i]
                tmp_out = tf.transpose(output1, [1, 0])
                t_out = tf.multiply(tmp_out,temporal_att[:,i])
                t_out = tf.transpose(t_out, [1, 0])              
                
            with tf.variable_scope("LSTM_CAP",reuse=reuse_flag):    
                output3, state3 = self.lstm_cap(tf.concat([padding, t_out], axis=1), state3)
                
        ##### Decoding #####
        for i in range(self.n_caption_lstm_step):
            tf.get_variable_scope()

            if i == 0:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
            
            with tf.variable_scope("LSTM_SEQ1", reuse=True):
                output1, state1 = self.lstm_seq1(padding, state1)
                
            with tf.variable_scope("LSTM_CAP", reuse=True):    
                output3, state3 = self.lstm_cap(tf.concat([current_embed, output1], axis=1), state3)
            
            logit_words = tf.nn.xw_plus_b(output3, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)
            
            
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, c3d_feat, generated_words, spatial_att, temporal_att, probs, embeds

    
