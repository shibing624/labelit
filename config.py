# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

input_file_path = "data/bjp.txt"
seg_input_file_path = "data/bjp_seg.txt"
col_sep = '\t'  # separate label and content of train data

# Active learning params
output_dir = "./output"  # Where to save outputs
sampling_method = "margin"  # Name of sampling method to use, can be any defined in AL_MAPPING in sampler.constants
warmstart_size = 0.02  # Float indicates percentage of training data to use in the initial warmstart model
batch_size = 3  # Can be float or integer.  Float indicates batch size as a percentage of training data size.
trials = 1  # Number of curves to create using different seeds
seed = 1  # Seed to use for rng and random state
confusions = 0.  # Percentage of labels to randomize
active_sampling_percentage = 1.0  # Mixture weights on active sampling.
score_method = "logistic"  # Method to use to calculate accuracy.
select_method = None  # Method to use for selecting points.
normalize_data = False  # Whether to normalize the data.
standardize_data = True  # Whether to standardize the data.
max_dataset_size = 15000  # maximum number of datapoints to include in data zero indicates no limit
train_horizon = 1.0  # how far to extend learning curve as a percent of train
do_save = True  # whether to save log and results

upper_thres = 0.85
lower_thres = 0.5
label_ratio = 0.9
label_min_num = 150
num_classes = 3

model_type = "logistic"
sentence_symbol_path = 'data/sentence_symbol.txt'
stop_words_path = 'data/stop_words.txt'

min_count = 1  # word will not be added to dictionary if it's frequency is less than min_count

# train type usage:  one of "tfidf_char, tfidf_word, tf_word",
feature_type = 'tf_word'
word_vocab_path = output_dir + "/vocab_" + feature_type + "_" + model_type + ".txt"  # vocab path
label_vocab_path = output_dir + "/label_" + feature_type + "_" + model_type + ".txt"  # label path
feature_vec_path = output_dir + "/vectorizer_" + feature_type + ".pkl"  # vector path
model_save_path = output_dir + "/model_" + feature_type + "_" + model_type + ".pkl"  # save model path

# predict
pred_save_path = output_dir + "/pred_result_" + feature_type + "_" + model_type + ".txt"  # predict data result

import os

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
