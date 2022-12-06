# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 智能标注
"""

import os
from time import time

import cleanlab
import numpy as np
from cleanlab.count import compute_confident_joint, estimate_cv_predicted_probabilities
from cleanlab.filter import find_label_issues
from cleanlab.classification import CleanLearning
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from loguru import logger
from labelit import config
from labelit.active_learning.choose_samples import ChooseSamples
from labelit.models.classic_model import get_model
from labelit.models.evaluate import eval
from labelit.models.feature import Feature
from labelit.preprocess import seg_data
from labelit.utils.data_utils import dump_pkl, write_vocab, build_vocab, load_vocab, data_reader, save


class DataObject(object):
    """数据字段定义"""

    def __init__(
            self,
            index=0,
            original_text="",
            segment_text="",
            human_label="",
            machine_label="",
            prob=0.0,
            feature="",
            rule_word="",
            need_label=False,
    ):
        self.index = index  # index值
        self.original_text = original_text  # 原物料,未切词
        self.segment_text = segment_text  # 切词结果(切词/切字)
        self.human_label = human_label  # 人工标注结果
        self.machine_label = machine_label  # 机器预测标签
        self.prob = prob  # 预测标签的概率
        self.feature = feature  # 该条样本的特征值
        self.rule_word = rule_word  # 规则词
        self.need_label = need_label  # 是否需要标注

    def __repr__(self):
        return "index: %s, human_label: %s, machine_label: %s, prob: %f, original_text: %s" % (
            self.index, self.human_label, self.machine_label, self.prob, self.original_text)


class LabelModel(object):
    """
    Online Model for Label
    """

    def __init__(
            self,
            input_file_path=config.input_file_path,
            seg_input_file_path=config.seg_input_file_path,
            word_vocab_path=config.word_vocab_path,
            label_vocab_path=config.label_vocab_path,
            feature_vec_path=config.feature_vec_path,
            model_save_path=config.model_save_path,
            pred_save_path=config.pred_save_path,
            feature_type=config.feature_type,
            segment_type=config.segment_type,
            model_type=config.model_type,
            num_classes=config.num_classes,
            col_sep=config.col_sep,
            min_count=config.min_count,
            lower_thres=config.lower_thres,
            upper_thres=config.upper_thres,
            label_confidence_threshold=config.label_confidence_threshold,
            label_min_size=config.label_min_size,
            batch_size=config.batch_size,
            warmstart_size=config.warmstart_size,
            sentence_symbol_path=config.sentence_symbol_path,
            stop_words_path=config.stop_words_path,
    ):
        self.input_file_path = input_file_path
        self.seg_input_file_path = seg_input_file_path
        self.sentence_symbol_path = sentence_symbol_path
        self.stop_words_path = stop_words_path
        self.word_vocab_path = word_vocab_path if word_vocab_path else "word_vocab.txt"
        self.label_vocab_path = label_vocab_path if label_vocab_path else "label_vocab.txt"
        self.feature_vec_path = feature_vec_path if feature_vec_path else "feature_vec.pkl"
        self.model_save_path = model_save_path if model_save_path else "model.pkl"
        self.pred_save_path = pred_save_path if pred_save_path else "predict.txt"
        self.feature_type = feature_type
        self.segment_type = segment_type
        self.num_classes = num_classes
        self.col_sep = col_sep
        self.min_count = min_count
        self.lower_thres = lower_thres
        self.upper_thres = upper_thres
        self.label_confidence_threshold = label_confidence_threshold
        self.cl = CleanLearning(clf=LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs'), verbose=True)

        # 1. load segment data
        if not os.path.exists(self.seg_input_file_path):
            start_time = time()
            seg_data(self.input_file_path, self.seg_input_file_path, col_sep=self.col_sep,
                     stop_words_path=self.stop_words_path, segment_type=segment_type)
            logger.info("spend time: %s s" % (time() - start_time))
        self.seg_contents, self.data_lbl = data_reader(self.seg_input_file_path, self.col_sep)

        # 2. load original data
        self.content, _ = data_reader(self.input_file_path, self.col_sep)

        # 3. load feature
        word_lst = []
        for i in self.seg_contents:
            word_lst.extend(i.split())
        # word vocab
        self.word_vocab = build_vocab(word_lst, min_count=self.min_count, sort=True, lower=True)
        # save word vocab
        write_vocab(self.word_vocab, self.word_vocab_path)
        # label
        label_vocab = build_vocab(self.data_lbl)
        # save label vocab
        write_vocab(label_vocab, self.label_vocab_path)
        label_id = load_vocab(self.label_vocab_path)
        logger.info("label_id: %s" % label_id)
        self.set_label_id(label_id)
        self.id_label = {v: k for k, v in label_id.items()}
        logger.info('num_classes:%d' % self.num_classes)
        self.data_feature = self._get_feature(self.word_vocab)

        # 4. assemble sample DataObject
        self.samples = self._get_samples(self.data_feature)
        self.batch_num = batch_size if batch_size > 1 else batch_size * len(self.samples)
        self.warmstart_num = warmstart_size if warmstart_size > 1 else warmstart_size * len(self.samples)
        self.label_min_num = label_min_size if label_min_size > 1 else label_min_size * len(self.samples)

        # 5. init model
        self.model = get_model(model_type)
        self.model_trained = False

    def _get_feature(self, word_vocab):
        # 提取特征
        logger.info(f"feature_type: {self.feature_type}\nseg_contents: \n{self.seg_contents[:2]}")
        feature = Feature(
            data=self.seg_contents,
            feature_type=self.feature_type,
            segment_type=self.segment_type,
            feature_vec_path=self.feature_vec_path,
            word_vocab=word_vocab,
            sentence_symbol_path=self.sentence_symbol_path,
            stop_words_path=self.stop_words_path
        )
        # get data feature
        return feature.get_feature()

    def _get_samples(self, data_feature):
        samples = []
        for i, text in enumerate(self.content):
            human_label = self.data_lbl[i] if i < len(self.data_lbl) else ""
            prob = 1.0 if human_label else 0.0
            sample = DataObject(i, text, segment_text=self.seg_contents[i],
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

    def _split_labeled_unlabeled_samples(self):
        # split labeled data and unlabeled data
        labeled_sample_list = []
        unlabeled_sample_list = []
        for i in self.samples:
            if i.prob >= self.upper_thres:
                labeled_sample_list.append(i)
            else:
                unlabeled_sample_list.append(i)
        self.set_labeled_sample_num(len(labeled_sample_list))
        self.set_unlabeled_sample_num(len(unlabeled_sample_list))
        logger.info(f"labeled size: {len(labeled_sample_list)}; unlabeled size: {len(unlabeled_sample_list)}")
        return labeled_sample_list, unlabeled_sample_list

    def _split_labels_and_text(self, labeled_sample_list):
        labeled_data_label = [i.human_label for i in labeled_sample_list if i.human_label]
        labeled_data_feature = [i.feature.toarray().tolist()[0] for i in labeled_sample_list]
        X = np.array(labeled_data_feature)
        labels = np.array([self.label_id[i] for i in labeled_data_label])
        return X, labels

    def _find_noise(self, labeled_sample_list):
        # find noise
        X, labels = self._split_labels_and_text(labeled_sample_list)
        # 找出问题样本
        label_issues_df = self.cl.find_label_issues(X, labels=labels)
        ordered_label_errors = label_issues_df["is_label_issue"].values
        logger.debug('[find_noise] ordered_label_errors index: {}, size: {}'.format(
            label_issues_df, len(label_issues_df)))
        noise_samples = [labeled_sample_list[i] for i, issue in enumerate(ordered_label_errors) if issue]
        return noise_samples, label_issues_df

    def find_noise(self):
        """
        找出噪声数据
        """
        labeled_sample_list, unlabeled_sample_list = self._split_labeled_unlabeled_samples()
        self._train(labeled_sample_list)
        return self._find_noise(labeled_sample_list)

    def train_with_no_noise(self):
        labeled_sample_list, unlabeled_sample_list = self._split_labeled_unlabeled_samples()
        X, labels = self._split_labels_and_text(labeled_sample_list)
        X_train, X_val, y_train, y_val = train_test_split(csr_matrix(X), labels)
        # cleanlab trains a robust version of your model that works more reliably with noisy data.
        self.cl.fit(X_train, y_train)

        eval(self.cl, X_val, y_val)

        # A true data-centric AI package, cleanlab quantifies class-level issues and overall data quality, for any dataset.
        cleanlab.dataset.health_summary(labels, confident_joint=self.cl.confident_joint, verbose=True)

    def _train(self, labeled_sample_list):
        # get data feature
        labeled_data_label = [i.human_label if i.human_label else i.machine_label for i in labeled_sample_list]
        labeled_data_feature = [i.feature.toarray().tolist()[0] for i in labeled_sample_list]
        X_train, X_val, y_train, y_val = train_test_split(csr_matrix(np.array(labeled_data_feature)),
                                                          labeled_data_label)
        # fit
        self.model.fit(X_train, y_train)
        self.model_trained = True
        # save model
        dump_pkl(self.model, self.model_save_path, overwrite=True)
        # evaluate model
        eval(self.model, X_val, y_val)

    def _predict_unlabeled_data(self, unlabeled_sample_list, batch_id):
        if not self.model_trained:
            raise RuntimeError("model not fit.")
        machine_samples_list = []
        # 预测未标注数据集
        unlabeled_data_feature = [i.feature.toarray().tolist()[0] for i in unlabeled_sample_list]
        if not unlabeled_sample_list:
            return machine_samples_list
        pred_result = self.model.predict_proba(csr_matrix(np.array(unlabeled_data_feature)))
        pred_label_proba = [(self.id_label[prob.argmax()], prob.max()) for prob in pred_result]

        # save middle result
        pred_output = [self.id_label[prob.argmax()] + self.col_sep + str(prob.max()) for prob in pred_result]
        pred_save_path = self.pred_save_path[:-4] + '_batch_' + str(batch_id) + '.txt'
        logger.debug("save infer label and prob result to: %s" % pred_save_path)
        unlabeled_data_text = [i.original_text for i in unlabeled_sample_list]
        save(pred_output, ture_labels=None, pred_save_path=pred_save_path, data_set=unlabeled_data_text)

        assert len(unlabeled_sample_list) == len(pred_label_proba)
        for unlabeled_sample, label_prob in zip(unlabeled_sample_list, pred_label_proba):
            self.samples[unlabeled_sample.index].machine_label = label_prob[0]
            self.samples[unlabeled_sample.index].prob = label_prob[1]
            machine_samples_list.append(unlabeled_sample)
        return machine_samples_list

    def _save_best_model(self):
        """
        保存最佳模型，使用所有数据再次训练模型
        """
        # split labeled data and unlabeled data
        output = []
        contents = []
        seg_contents = []
        features = []
        labels = []
        for i in self.samples:
            label = i.human_label if i.human_label else i.machine_label
            output.append(label + self.col_sep + str(i.prob))
            seg_contents.append(i.segment_text)
            contents.append(i.original_text)
            labels.append(label)
            features.append(i.feature.toarray().tolist()[0])
        # get data feature
        X_train, X_val, y_train, y_val = train_test_split(csr_matrix(np.array(features)), labels)

        # fit
        self.model.fit(X_train, y_train)

        # save model
        dump_pkl(self.model, self.model_save_path, overwrite=True)
        eval(self.model, X_val, y_val)
        save(output, ture_labels=None, pred_save_path=self.pred_save_path, data_set=contents)

    def _check_model_can_start(self, labeled_samples_list):
        """
        根据识别出的标签量, 判断模型是否达到开始训练要求
        :param labeled_samples_list: [DataObject], 人工标注结果
        :return: False, 不可以训练; True, 可以开始训练
        """
        human_labels = [i.human_label for i in labeled_samples_list]
        human_label_size = len(set(human_labels))
        logger.info(f'human label classes: {human_label_size}; set num_classes: {self.num_classes}')
        assert human_label_size == self.num_classes, "human label type need same as num classes."
        labeled_type_num = dict()
        for i in set(human_labels):
            count = 0
            for j in human_labels:
                if j == i:
                    count += 1
            labeled_type_num[i] = count

        for k, v in labeled_type_num.items():
            if v < self.warmstart_num:
                return False
        return True

    def _check_model_can_finish(self, machine_samples_list):
        """
        根据识别出的标签量, 判断模型是否达到结束要求
        :param machine_samples_list: [DataObject], 机器预测结果
        :return: False, 需要继续迭代; True, 可以结束
        """
        is_finish = False
        trusted_index, untrusted_index = ChooseSamples.split_by_threshold(machine_samples_list,
                                                                          self.lower_thres,
                                                                          self.upper_thres)
        logger.debug("[check model can finish] trusted_index samples:%d; untrusted_index samples:%d"
                     % (len(trusted_index), len(untrusted_index)))
        p = 1 - (len(untrusted_index) + 0.0) / len(self.samples)
        if p >= self.label_confidence_threshold and self.get_labeled_sample_num() > self.label_min_num:
            is_finish = True
        logger.debug("[check model can finish] is_finish:%s, trusted label rate:%f; label_confidence_threshold:%f"
                     % (is_finish, p, self.label_confidence_threshold))
        return is_finish

    def _input_human_label(self, choose_sample):
        for i, sample in enumerate(choose_sample):
            print("[batch] id:%d, [sample] %s" % (i, sample))
            print("id_label:%s" % self.id_label)
            # 检测输入标签
            while True:
                input_label_id = input("input label id:").lower().strip()
                if input_label_id in ['q', 'e', 's', 'quit', 'exit', 'stop']:
                    logger.warning('process stop.')
                    exit(1)
                if input_label_id.isdigit() and (int(input_label_id) in self.id_label):
                    break
            label = self.id_label[int(input_label_id)]
            # set human label for sample
            self.samples[sample.index].human_label = label
            self.samples[sample.index].prob = 1.0
            self.samples[sample.index].machine_label = ""

    def start_label(self):
        batch_id = 0
        while True:
            labeled_sample_list, unlabeled_sample_list = self._split_labeled_unlabeled_samples()
            # 启动自动标注
            if batch_id == 0:
                if not self._check_model_can_start(labeled_sample_list):
                    # 训练模型的启动标注样本量不足，随机抽样补充标注样本
                    choose_sample = ChooseSamples.choose_random(unlabeled_sample_list, self.batch_num)
                    self._input_human_label(choose_sample)
                else:
                    # train and evaluate model first
                    self._train(labeled_sample_list)
                    # 启动样本量足够，检测噪声数据再次标注（精标）
                    noise_samples, label_issues_df = self._find_noise(labeled_sample_list)
                    self._input_human_label(noise_samples)

            # train and evaluate model after delete noise
            self._train(labeled_sample_list)
            # predict unlabeled data
            machine_samples_list = self._predict_unlabeled_data(unlabeled_sample_list, batch_id)
            if self._check_model_can_finish(machine_samples_list):
                noise_samples, label_issues_df = self._find_noise(labeled_sample_list)
                self._input_human_label(noise_samples)
                self._save_best_model()
                break
            choose_sample = ChooseSamples.choose_label_data_random(
                machine_samples_list,
                self.batch_num,
                self.lower_thres,
                self.upper_thres,
                self.label_id
            )
            self._input_human_label(choose_sample)
            batch_id += 1
