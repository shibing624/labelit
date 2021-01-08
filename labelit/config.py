# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

input_file_path = os.path.join(pwd_path, "../extra_data/samples.txt")

# segment_type optionals: "word, char"
segment_type = 'word'
# feature_type optionals: "tfidf, tf"
feature_type = 'tfidf'

seg_input_file_path = os.path.join(pwd_path, "../extra_data/samples_seg_{}.txt".format(segment_type))
col_sep = '\t'  # separate label and content of train data
num_classes = 6

# active learning params
output_dir = os.path.join(pwd_path, "../extra_data")  # where to save outputs
sampling_method = "margin"  # name of sampling method to use, can be any defined in AL_MAPPING in sampler.constants
warmstart_size = 20  # float indicates percentage of training data to use in the initial warmstart model
batch_size = 5  # can be float or integer.  Float indicates batch size as a percentage of training data size.
score_method = "logistic"  # method to use to calculate accuracy.

upper_thres = 0.9  # upper threshold of label
lower_thres = 0.3  # lower threshold of label

label_confidence_threshold = 0.9  # label min confidence threshold
label_min_size = 0.2  # num of labeled sample, for finish mark process

model_type = "logistic"
sentence_symbol_path = os.path.join(pwd_path, "data/sentence_symbol.txt")
stop_words_path = os.path.join(pwd_path, "data/stop_words.txt")
min_count = 1  # word will not be added to dictionary if it's frequency is less than min_count

# vocab path
word_vocab_path = os.path.join(output_dir, "vocab_{}_{}_{}.txt".format(feature_type, segment_type, model_type))
# label path
label_vocab_path = os.path.join(output_dir, "label_{}_{}_{}.txt".format(feature_type, segment_type, model_type))
# feature vector path
feature_vec_path = os.path.join(output_dir, "feature_{}_{}.pkl".format(feature_type, segment_type))
# save model path
model_save_path = os.path.join(output_dir, "model_{}_{}_{}.pkl".format(feature_type, segment_type, model_type))
# predict result
pred_save_path = os.path.join(output_dir, "pred_result_{}_{}_{}.txt".format(feature_type, segment_type, model_type))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
