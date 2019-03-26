# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 智能标注
"""

import os
from time import time

from sklearn.model_selection import train_test_split

from active_learning.choose_samples import ChooseSamples
from models.classic_model import get_model
from models.evaluate import eval
from models.feature import Feature
from preprocess import seg_data
from utils.data_utils import dump_pkl, write_vocab, build_vocab, load_vocab, data_reader, save
from utils.io_utils import get_logger

logger = get_logger(__name__)


class DataObject(object):
    """数据字段定义"""

    def __init__(self, id=0, original_text="", seg_text_word="", seg_text_char="", human_label="", machine_label="",
                 prob=0.0, feature="", rule_word="", need_label=False):
        self.id = id
        self.original_text = original_text  # 原物料,未切词
        self.seg_text_word = seg_text_word  # 切词结果(混排)
        self.seg_text_char = seg_text_char  # 切字结果
        self.human_label = human_label  # 人工标注结果
        self.machine_label = machine_label  # 机器预测标签
        self.prob = prob  # 预测标签的概率
        self.feature = feature  # 该条样本的特征值
        self.rule_word = rule_word  # 规则词
        self.need_label = need_label  # 是否需要标注

    def __repr__(self):
        return "id:" + str(self.id) + "\toriginal_text:" + self.original_text


class LabelModel(object):
    """
    Online Model for Label
    """

    def __init__(self, input_file_path,
                 seg_input_file_path='',
                 word_vocab_path='',
                 label_vocab_path='',
                 feature_vec_path='',
                 model_save_path='',
                 pred_save_path='',
                 feature_type='tf_word',
                 model_type='logistic',
                 num_classes=2,
                 col_sep='\t',
                 min_count=1,
                 lower_thres=0.5,
                 upper_thres=0.85,
                 label_ratio=0.9,
                 label_min_num=200,
                 batch_size=10,
                 stop_words_path='data/stop_words.txt'):
        self.input_file_path = input_file_path
        self.seg_input_file_path = seg_input_file_path
        self.stop_words_path = stop_words_path
        self.word_vocab_path = word_vocab_path
        self.label_vocab_path = label_vocab_path
        self.feature_vec_path = feature_vec_path
        self.model_save_path = model_save_path
        self.pred_save_path = pred_save_path
        self.feature_type = feature_type
        self.model_type = model_type
        self.num_classes = num_classes
        self.col_sep = col_sep
        self.min_count = min_count
        self.lower_thres = lower_thres
        self.upper_thres = upper_thres
        self.label_ratio = label_ratio
        self.label_min_num = label_min_num

        # 1. load segment data
        if not os.path.exists(self.seg_input_file_path):
            start_time = time()
            seg_data(self.input_file_path, self.seg_input_file_path, col_sep=self.col_sep,
                     stop_words_path=self.stop_words_path)
            logger.info("spend time: %s s" % (time() - start_time))
        self.seg_contents, self.data_lbl = data_reader(self.seg_input_file_path, self.col_sep)

        # 2. load original data
        self.content, _ = data_reader(self.input_file_path, self.col_sep)

        # 3. load feature
        word_lst = []
        for i in self.seg_contents:
            word_lst.extend(i.split())
        # word vocab
        word_vocab = build_vocab(word_lst, min_count=self.min_count, sort=True, lower=True)
        # save word vocab
        write_vocab(word_vocab, self.word_vocab_path)
        # label
        label_vocab = build_vocab(self.data_lbl)
        # save label vocab
        write_vocab(label_vocab, self.label_vocab_path)
        label_id = load_vocab(self.label_vocab_path)
        print(label_id)
        self.set_label_id(label_id)
        self.id_label = {v: k for k, v in label_id.items()}
        print(label_vocab)
        data_label = [label_id[i] for i in self.data_lbl]
        print('num_classes:%d' % self.num_classes)
        self.data_feature = self._get_feature(word_vocab)

        # 4. assemble sample DataObject
        self.samples = self._get_samples(self.data_feature)
        self.batch_num = batch_size if batch_size > 1 else batch_size * len(self.samples)

    def _get_feature(self, word_vocab):
        # 提取特征
        print(self.feature_type)
        print("seg_contents:")
        print(self.seg_contents[:2])
        feature = Feature(data=self.seg_contents, feature_type=self.feature_type,
                          feature_vec_path=self.feature_vec_path, word_vocab=word_vocab)
        # get data feature
        return feature.get_feature()

    def _get_samples(self, data_feature):
        samples = []
        for i, text in enumerate(self.content):
            human_label = self.data_lbl[i] if i < len(self.data_lbl) else ""
            prob = 1.0 if human_label else 0.0
            sample = DataObject(i, text, seg_text_word=self.seg_contents[i], seg_text_char=' '.join(list(text)),
                                human_label=human_label, prob=prob, feature=data_feature[i])
            samples.append(sample)
        return samples

    def set_feature_id(self, feature_id):
        self.feature_id = feature_id

    def get_feature_id(self):
        return self.feature_id

    def set_label_id(self, label_id):
        self.label_id = label_id

    def get_label_id(self):
        return self.label_id

    def set_labeled_sample_num(self, labeled_sample_num):
        self.labeled_sample_num = labeled_sample_num

    def get_labeled_sample_num(self):
        return self.labeled_sample_num

    def set_unlabeled_sample_num(self, unlabeled_sample_num):
        self.unlabeled_sample_num = unlabeled_sample_num

    def get_unlabeled_sample_num(self):
        return self.unlabeled_sample_num

    def train(self, samples, upper_thres, model_type, pred_save_path, feature_vec_path,
              num_classes, model_save_path, feature_type, label_id, col_sep):
        # split labeled data and unlabeled data
        labeled_sample_list = []
        unlabeled_sample_list = []
        labeled_data_label = []
        labeled_data_content = []
        unlabeled_data_content = []
        for i in samples:
            if i.human_label or i.prob > upper_thres:
                labeled_sample_list.append(i)
                labeled_data_content.append(i.seg_text_word)
                lbl = i.human_label if i.human_label else i.machine_label
                labeled_data_label.append(lbl)
            else:
                unlabeled_sample_list.append(i)
                unlabeled_data_content.append(i.seg_text_word)
        print("labeled size: %d" % len(labeled_sample_list))
        self.set_labeled_sample_num(len(labeled_sample_list))
        print("unlabeled size: %d" % len(unlabeled_sample_list))
        self.set_unlabeled_sample_num(len(unlabeled_sample_list))
        # get data feature
        labeled_data_feature = Feature(data=labeled_data_content, feature_type=feature_type,
                                       feature_vec_path=feature_vec_path, is_infer=True).get_feature()
        X_train, X_val, y_train, y_val = train_test_split(labeled_data_feature, labeled_data_label)
        model = get_model(model_type)
        # fit
        model.fit(X_train, y_train)

        # save model
        dump_pkl(model, model_save_path, overwrite=True)

        eval(model, X_val, y_val, num_classes=num_classes)

        # 预测未标注数据集
        unlabeled_data_feature = Feature(data=unlabeled_data_content, feature_type=feature_type,
                                         feature_vec_path=feature_vec_path, is_infer=True).get_feature()

        # predict
        pred_label_probs = model.predict_proba(unlabeled_data_feature)

        # label id map
        id_label = {v: k for k, v in label_id.items()}

        pred_label_proba = [(id_label[prob.argmax()], prob.max()) for prob in pred_label_probs]
        print(pred_label_proba[0])

        # save middle result
        pred_output = [id_label[prob.argmax()] + col_sep + str(prob.max()) for prob in pred_label_probs]
        logger.info("save infer label and prob result to: %s" % pred_save_path)
        save(pred_output, ture_labels=None, pred_save_path=pred_save_path)

        machine_samples_list = []
        assert len(unlabeled_sample_list) == len(pred_label_proba)
        for unlabeled_sample, label_prob in zip(unlabeled_sample_list, pred_label_proba):
            unlabeled_sample.machine_label = label_prob[0]
            unlabeled_sample.prob = label_prob[1]
            machine_samples_list.append(unlabeled_sample)
        return machine_samples_list

    def check_model_can_finish(self, samples, machine_samples_list):
        """
        根据识别出的标签量, 判断模型是否达到结束要求
        :param samples: [DataObject], 所有预测的数据
        :param machine_samples_list: [DataObject], 机器预测结果
        :return: False, 需要继续迭代; True, 可以结束
        """
        flag = False
        out_index, in_index = ChooseSamples.split_by_thres(machine_samples_list, self.lower_thres,
                                                           self.upper_thres)
        p = 1 - (len(in_index) + 0.0) / len(samples)
        if p >= self.label_ratio and self.get_labeled_sample_num() > self.label_min_num:
            flag = True
        return flag

    def input_human_label(self, choose_sample, samples, id_label):
        for i, sample in enumerate(choose_sample):
            print("batch id:%d" % i)
            print(sample)
            idx = sample.id

            print("id_label:%s" % id_label)
            # 检测输入标签
            while True:
                input_label_id = input("input label id:").strip()
                if input_label_id.isdigit() and (int(input_label_id) in id_label):
                    break
            label = id_label[int(input_label_id)]
            samples[idx].human_label = label
            samples[idx].prob = 1.0
            samples[idx].machine_label = ""

    def label(self):
        while True:
            machine_samples_list = self.train(self.samples, self.upper_thres, self.model_type, self.pred_save_path,
                                              self.feature_vec_path,
                                              self.num_classes, self.model_save_path, self.feature_type, self.label_id,
                                              self.col_sep)
            choose_sample = ChooseSamples.choose_label_data_random(machine_samples_list, self.batch_num,
                                                                   self.lower_thres, self.upper_thres, self.label_id)
            if self.check_model_can_finish(self.samples, machine_samples_list):
                break
            self.input_human_label(choose_sample, self.samples, self.id_label)


if __name__ == "__main__":
    import config

    lm = LabelModel(config.input_file_path,
                    config.seg_input_file_path,
                    config.word_vocab_path,
                    config.label_vocab_path,
                    config.feature_vec_path,
                    config.model_save_path,
                    config.pred_save_path,
                    feature_type=config.feature_type,
                    model_type=config.model_type,
                    num_classes=config.num_classes,
                    col_sep=config.col_sep,
                    min_count=config.min_count,
                    lower_thres=config.lower_thres,
                    upper_thres=config.upper_thres,
                    label_ratio=config.label_ratio,
                    label_min_num=config.label_min_num,
                    batch_size=config.batch_size)
    lm.label()
