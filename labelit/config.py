# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

input_file_path = os.path.join(pwd_path, "../extra_data/bjp.txt")
seg_input_file_path = os.path.join(pwd_path, "../extra_data/bjp_seg.txt")
col_sep = '\t'  # separate label and content of train data
num_classes = 3

# Active learning params
output_dir = os.path.join(pwd_path, "../extra_data")  # Where to save outputs
sampling_method = "margin"  # Name of sampling method to use, can be any defined in AL_MAPPING in sampler.constants
warmstart_size = 20  # Float indicates percentage of training data to use in the initial warmstart model
batch_size = 6  # Can be float or integer.  Float indicates batch size as a percentage of training data size.
score_method = "logistic"  # Method to use to calculate accuracy.

upper_thres = 0.9
lower_thres = 0.6

label_ratio = 0.9
label_min_size = 0.2

model_type = "logistic"
sentence_symbol_path = os.path.join(pwd_path, "data/sentence_symbol.txt")
stop_words_path = os.path.join(pwd_path, "data/stop_words.txt")

min_count = 1  # word will not be added to dictionary if it's frequency is less than min_count

# train type usage:  one of "tfidf_char, tfidf_word, tf_word",
feature_type = 'tfidf_char'
word_vocab_path = os.path.join(output_dir, "vocab_{}_{}.txt".format(feature_type, model_type))  # vocab path
label_vocab_path = os.path.join(output_dir, "label_{}_{}.txt".format(feature_type, model_type))  # label path
feature_vec_path = os.path.join(output_dir, "feature_{}.pkl".format(feature_type))  # feature vector path
model_save_path = os.path.join(output_dir, "model_{}_{}.pkl".format(feature_type, model_type))  # save model path

# predict result
pred_save_path = os.path.join(output_dir, "pred_result_{}_{}.txt".format(feature_type, model_type))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
