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
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_step, n_video_lstm_step, n_caption_lstm_step, n_attribute_category, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        
        self.n_lstm_step = n_lstm_step
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step
        self.n_attribute_category = n_attribute_category
        
        #LSTM for language modeling
        self.lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        
        self.lstm_seq1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm_seq2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        
        with tf.device("cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')        
        
        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
        
        #image feature
    
    #Generate fully connected layer
    def fc(self, input, output_shape, activation_fn=tf.nn.relu, name="fc"):
        output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
        return output            
    
#    #g_theta MLP in relational network
#    def g_theta(self, o_i, o_j, scope='g_theta'):#, reuse=True):
#        #with tf.variable_scope(scope, reuse=reuse) as scope:
#            #if not reuse: log.warn(scope.name)
#            reshape_input = tf.reshape(tf.concat([o_i, o_j], axis=2), [self.batch_size * self.n_lstm_step, self.dim_hidden + self.n_attribute_category])
#            g_1 = self.fc(reshape_input, 256, name='g_1')
#            g_2 = self.fc(g_1, 256, name='g_2')
#            g_3 = self.fc(g_2, 256, name='g_3')
#            g_4 = self.fc(g_3, 256, name='g_4')
#            return g_4
#
#    #f_phi MLP in relational network
#    def f_phi(self, g, scope='f_phi'):
#        #with tf.variable_scope(scope) as scope:
#            #log.warn(scope.name)
#            return fc_fin

    
    
    #build_model for training
   
    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step, self.dim_image]) 
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step]) 
        
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])
        
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b) # batch_size * n_lstm_step, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_step, self.dim_hidden])
        #image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h
        
        state1 = tf.zeros([self.batch_size, self.lstm_seq1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm_seq2.state_size])        
        state3 = tf.zeros([self.batch_size, self.lstm_cap.state_size])        
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        ##attribute information
        spatial_att = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step, self.n_attribute_category]) 
        temporal_att = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_step]) 

        probs = []
        loss = 0.0
        
        t_out = tf.placeholder(tf.float32, [self.batch_size, self.dim_hidden])
        
        ########## Model setup ##########
        ##### without for loop ####
        reshape_input = tf.reshape(tf.concat([image_emb, spatial_att], axis=2), [self.batch_size * self.n_lstm_step, self.dim_hidden + self.n_attribute_category])
        
        #G theta
        g_1 = self.fc(reshape_input, 256, name='g_1')
        g_2 = self.fc(g_1, 256, name='g_2')
        g_3 = self.fc(g_2, 256, name='g_3')
        g_4 = self.fc(g_3, 256, name='g_4')        
        #elementwise sum is not implemented
        g_out = g_4
        #F_phi
        fc_1 = self.fc(g_out, 256, name='fc_1')
        fc_2 = self.fc(fc_1, 256, name='fc_2')
        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=True)
        fc_3 = self.fc(fc_2, 256, activation_fn=None, name='fc_3')
        f_out = tf.reshape(fc_3, [self.batch_size, self.n_lstm_step, 256])
                

        #batch_size step input_dim
#        rnn.static_rnn(lstm_seq1,f_out,dtype=tf.float32)
        
#        output1, state1 = self.lstm_seq1(f_out, state1)
       ##### Encoding #####
        for i in range(self.n_video_lstm_step):
            #final_frame_feat = tf.concat([image_emb, spatial_att])
            #g1 = g_theta(final_frame_feat)
            #g_out = self.g_theta(image_emb[:,i,:], spatial_att[:,i,:])
            #f_out = self.f_phi(g_out)
#            t_out = f_out * temporal_att[i]
#
#            output1, state1 = self.lstm_seq1(f_out, state1)
#            t_out = output1 * temporal_att[i]
#            output2, state2 = self.lstm_seq2(t_out, state2)
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
                    
            with tf.variable_scope("LSTM_SEQ2",reuse=reuse_flag):
                output2, state2 = self.lstm_seq2(t_out, state2)
                
                
            with tf.variable_scope("LSTM_CAP",reuse=reuse_flag):    
                output3, state3 = self.lstm_cap(tf.concat([padding, output2], axis=1), state3)
                
        ##### Decoding #####
        for i in range(self.n_caption_lstm_step):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])            
            
            tf.get_variable_scope()
            with tf.variable_scope("LSTM_SEQ2", reuse=True):
                output2, state2 = self.lstm_seq2(padding, state2)
                
            with tf.variable_scope("LSTM_CAP", reuse=True):    
                output3, state3 = self.lstm_cap(tf.concat([current_embed, output2], axis=1), state3)

                
            labels = tf.expand_dims(caption[:,i]+1, 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], axis=1)
            outshape = tf.stack([self.batch_size, self.n_words])
            onehot_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output3, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)
            loss += current_loss

        return loss, video, video_mask, caption, caption_mask, spatial_att, temporal_att, probs

    #build_generator for test
    #def build_generator(self):
        





























                