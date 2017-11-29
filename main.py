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
from util_functions import *

def get_validation_loss(sess, current_val_data, wordtoix, s_att, t_att, att_name, 
                        tf_loss, tf_video, tf_c3d_feat, tf_caption, tf_caption_mask, tf_spatial_att, tf_temporal_att):
    val_data = current_val_data
    val_captions = val_data['caption'].values
    val_captions = caption_parsing(captions = val_captions)
    
    loss_on_validation = []

    for start, end in zip(
            range(0, len(val_data), batch_size),
            range(batch_size, len(val_data), batch_size)):
        
        current_batch = val_data[start:end]
        
        #Visual Feature load: Inception-Resnet
        current_vis_feats_path = current_batch['vis_feat_path'].values            
        current_vis_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
        current_vis_feats_vals = map(lambda vid: np.load(vid), current_vis_feats_path)
    

        for ind, feat in enumerate(current_vis_feats_vals):
            #feat = np.reshape(feat, [n_video_lstm_step, dim_image])
            current_vis_feats[ind][:len(current_vis_feats_vals[ind])] = feat


        current_c3d_feats_path = current_batch['c3d_feat_path'].values
        current_c3d_feats = np.zeros((batch_size, length_c3d, dim_c3d))
        current_c3d_feats_val = map(lambda vid: np.load(vid), current_c3d_feats_path)

        for ind, feat in enumerate(current_c3d_feats_val):
            feat = np.reshape(feat, [length_c3d, dim_c3d])
            current_c3d_feats[ind][:len(current_c3d_feats_val[ind])] = feat

        
        current_s_att, current_t_att = get_att_batch(current_vis_feats_path, batch_size, s_att, t_att, att_name)

        
        current_captions = current_batch['caption'].values
        current_captions = caption_parsing(captions = current_captions)
        
        
        current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:n_caption_lstm_step - 1]
                                                   if word in wordtoix], current_captions)
    
        current_caption_matrix = np.zeros((batch_size, n_caption_lstm_step))
        current_caption_masks = np.zeros((batch_size, n_caption_lstm_step))
        for ind, row in enumerate(current_caption_masks):
            valid_length = len(current_caption_ind[ind])
            row[:valid_length+1] = 1
            current_caption_matrix[ind, :valid_length] = current_caption_ind[ind]

        
        
        val_loss = sess.run(tf_loss, feed_dict={
                tf_video: current_vis_feats,
                tf_c3d_feat: current_c3d_feats,
                tf_caption: current_caption_matrix,
                tf_caption_mask: current_caption_masks,
                tf_spatial_att: current_s_att,
                tf_temporal_att: current_t_att
                })

        loss_on_validation.append(val_loss)
        #print 'idx: ', start, " Epoch: ", epoch, " loss: " , loss_val, ' Elapsed time: ', str((time.time() - start_time))
        
        
    return np.mean(loss_on_validation)

def train():
    #=======================================================================================
    print 'Load Captions'
    train_data, val_data, _ = get_video_data(tr_gt_path, ts_gt_path, tr_visf_path, ts_visf_path, tr_c3d_path, ts_c3d_path)
    captions = train_data['caption'].values
    captions = caption_parsing(captions)

    print 'Preprocessing words'
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=5)

    #=======================================================================================
    print 'Load attributes'
    #s_att, t_att, att_name = load_attribute(tr_video_path, tr_att_path, tr_visf_path, True)
        
    s_att = np.load("./data/tr_s_attention.npy")
    t_att = np.load("./data/tr_t_attention.npy")
    att_name = np.load("./data/tr_att_name.npy")

    #=======================================================================================
    
    print "Generate Model\n"
    model = Video_Caption_Generator(
            dim_image = dim_image, #attribute set
            n_words = len(wordtoix),
            dim_hidden = dim_hidden,
            dim_c3d = dim_c3d,
            length_c3d = length_c3d,
            c3d_space = c3d_space,
            batch_size = batch_size,
            n_lstm_step = n_frame_step,
            n_video_lstm_step = n_video_lstm_step,
            n_caption_lstm_step = n_caption_lstm_step,
            n_attribute_category = n_attribute_category,
            bias_init_vector = bias_init_vector)    
    print "Generate Model Done\n"
    
    print "Build Model\n"
    tf_loss, tf_video, tf_c3d_feat, tf_caption, tf_caption_mask, tf_spatial_att, tf_temporal_att, tf_probs =  model.build_model()    
    print "Build Model Done\n"
    
    sess = tf.InteractiveSession()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    saver = tf.train.Saver(max_to_keep = 400)
    tf.global_variables_initializer().run()    

    
    loss_to_draw = []
    loss_to_draw_val = []
    model_counter = 0

    for epoch in range(0, n_epochs):
        loss_to_draw_train = []
        print "Epoch: " + str(epoch)
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]
        current_train_data = train_data

        saving_schedule = []
        step_size =  (int(len(current_train_data) * save_every_n_epoch) // batch_size ) * batch_size
        saving_schedule = range(0, len(current_train_data) - step_size, step_size)
        print saving_schedule
        
        n_step = np.shape(saving_schedule)[0]
        step_count = 0
        step_summary = np.zeros([n_step])        
    
        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):
            start_time = time.time()
            
            current_batch = current_train_data[start:end]
            
            #Visual Feature load: Inception-Resnet
            current_vis_feats_path = current_batch['vis_feat_path'].values            
            current_vis_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_vis_feats_vals = map(lambda vid: np.load(vid), current_vis_feats_path)
            
            for ind, feat in enumerate(current_vis_feats_vals):
                feat = np.reshape(feat, [n_video_lstm_step, dim_image])
                current_vis_feats[ind][:len(current_vis_feats_vals[ind])] = feat

            current_c3d_feats_path = current_batch['c3d_feat_path'].values
            current_c3d_feats = np.zeros((batch_size, length_c3d, dim_c3d ))
            current_c3d_feats_val = map(lambda vid: np.load(vid), current_c3d_feats_path)

            for ind2, feat2 in enumerate(current_c3d_feats_val):
                feat2 = np.reshape(feat2, [length_c3d, dim_c3d])
                current_c3d_feats[ind2][:len(current_c3d_feats_val[ind])] = feat2

            ###Attributes###
            current_s_att, current_t_att = get_att_batch(current_vis_feats_path, batch_size, s_att, t_att, att_name)            
            
            ###Captions###    
            current_captions = current_batch['caption'].values
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:n_caption_lstm_step - 1]
                                                   if word in wordtoix], current_captions)

            current_caption_matrix = np.zeros((batch_size, n_caption_lstm_step))
            current_caption_masks = np.zeros((batch_size, n_caption_lstm_step))

            for ind, row in enumerate(current_caption_masks):
                valid_length = len(current_caption_ind[ind])
                row[:valid_length+1] = 1
                current_caption_matrix[ind, :valid_length] = current_caption_ind[ind]
            
            _, train_loss = sess.run([train_op, tf_loss], feed_dict={
                    tf_video: current_vis_feats,
                    tf_c3d_feat: current_c3d_feats,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks,
                    tf_spatial_att: current_s_att,
                    tf_temporal_att: current_t_att
                    })

            loss_to_draw_train.append(train_loss)

                        
            if start in saving_schedule:
                print start
                train_loss = np.mean(loss_to_draw_train[-5:])
                val_loss = get_validation_loss(sess, val_data, wordtoix, s_att, t_att, att_name, tf_loss, tf_video,
                                       tf_c3d_feat, tf_caption, tf_caption_mask, tf_spatial_att, tf_temporal_att)
                #print model loss
                step_summary = print_remaining_time(epoch, start_time, step_summary, step_count, n_step, val_loss, train_loss)
                
                #Draw loss image
                loss_to_draw.append(np.mean(loss_to_draw_train))
                loss_to_draw_val.append(val_loss)
                draw_loss_graph(epoch, step_count, model_counter, loss_to_draw, loss_to_draw_val)
                
                #save model
                print "Epoch ", epoch + 1, " step ", step_count, " is done. Saving the model ..."
                sys.stdout.flush()
                saver.save(sess, os.path.join(model_path, 'model'), global_step=model_counter)

                model_counter +=1
                step_count += 1


def test(model_path):
    scorer = COCOScorer()

    _, _, test_data = get_video_data(tr_gt_path, ts_gt_path, tr_visf_path, ts_visf_path, tr_c3d_path, ts_c3d_path)

    test_videos = test_data['vis_feat_path'].unique()
    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())
    bias_init_vector = np.load('./data/bias_init_vector.npy')

    #s_att, t_att, att_name = load_attribute(ts_video_path, ts_att_path, ts_visf_path, False)
    s_att = np.load("./data/ts_s_attention.npy")
    t_att = np.load("./data/ts_t_attention.npy")
    att_name = np.load("./data/ts_att_name.npy")


    print "Generate Model\n"
    model = Video_Caption_Generator(
            dim_image = dim_image, #attribute set
            n_words = len(ixtoword),
            dim_hidden = dim_hidden,
            dim_c3d = dim_c3d,
            length_c3d = length_c3d,
            c3d_space = c3d_space,
            batch_size = ts_batch_size,
            n_lstm_step = n_frame_step,
            n_video_lstm_step = n_video_lstm_step,
            n_caption_lstm_step = n_caption_lstm_step,
            n_attribute_category = n_attribute_category,
            bias_init_vector = bias_init_vector)    
    print "Generate Model Done\n"

    print "Build Model\n"
    tf_video, tf_c3d, tf_caption, tf_spatial_att, tf_temporal_att, tf_probs, last_embed_tf =  model.build_generator()    
    print "Build Model Done\n"
    
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    #test_output_txt_fd = open(output_txt_fname, 'w')
    
    
    splits = []
    splits.append((test_data['vis_feat_path'].unique(), test_data))
    
    results = []
    vid_count = 0
    for split, gt_dataframe in splits:
        gts = convert_data_to_coco_scorer_format(gt_dataframe)
        samples = {}
        
        #for idx in range() test_data
        
        for idx, vis_feat_path in enumerate(split):
            print idx, vis_feat_path
            video_feat = np.load(vis_feat_path)[None,...]
            #video_feat = np.reshape(video_feat, [1, n_video_lstm_step, dim_image])
            
            #load c3d feat
            videoID = vis_feat_path[len(ts_visf_path)+1:]
            c3d_path = os.path.join(ts_c3d_path,videoID)
            c3d_feat = np.load(c3d_path)
            c3d_feat = np.reshape(c3d_feat, [1, length_c3d, dim_c3d])
            #find attributes
            idx = np.where(att_name == vis_feat_path)[0][0]
            current_s_att = np.reshape(s_att[idx], [1, n_video_lstm_step, n_attribute_category])
            current_t_att = np.reshape(t_att[idx], [1, n_video_lstm_step])


            generated_word_index = sess.run(tf_caption, feed_dict={
                    tf_video:video_feat, 
                    tf_c3d: c3d_feat,
                    tf_spatial_att:current_s_att,
                    tf_temporal_att:current_t_att
                })

    
            generated_words = ixtoword[generated_word_index]
            
            punctuation = np.argmax(np.array(generated_words) == '.')+1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            print generated_sentence,'\n'
            
            video_id = vis_feat_path.split("/")[-1].split(".")[0] #+ ".jpg"
            samples[video_id] = [{u'image_id': video_id, u'caption': generated_sentence}]
            vid_count +=1
            print str(vid_count)+"/2990: " + video_id + ": " + generated_sentence,'\n'
                
        print "Calculating Score..."
        with suppress_stdout_stderr():
            valid_score = scorer.score(gts, samples, samples.keys())
        results.append(valid_score)
        print valid_score


if __name__ == '__main__':
    #Training Model
    train()
#    
##    #Test Model
    model_num = 1 #158981378
#    
    saved_model = model_path + "/model-" + str(model_num)
    test(model_path = saved_model)
    #test(model_path = saved_model, output_txt_fname = output_txt)
    
    
