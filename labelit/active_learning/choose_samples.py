# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import math
import random
from loguru import logger


class ChooseSamples(object):
    """ChooseSamples
    """

    @staticmethod
    def choose_random(machine_samples_list, batch_num):
        """
        随机抽取数据, batch_num个, 模型不可用时, 会调用这个抽样方法
        :param machine_samples_list: [DataObject], 机器识别的数据集
        :param batch_num: 需要抽取的数量
        :return: 在machine_samples_list中的抽样数据, list [DataObject]
        """
        choose = machine_samples_list if len(machine_samples_list) < batch_num else random.sample(machine_samples_list,
                                                                                                  batch_num)
        logger.info("choose_sample size：%d" % len(choose))
        return choose

    @staticmethod
    def choose_label_data_random(machine_samples_list, batch_num, lower_thres, upper_thres, label_id):
        """
        随机抽取阈值内的数据, batch_num个, 不足的用阈值外数据补全
        :param machine_samples_list: [DataObject], 机器识别的数据集
        :param batch_num: 需要抽取的个数
        :param lower_thres: 下界
        :param upper_thres: 上界
        :param label_id: label
        :return: 在machine_samples_list中的抽样数据, list [DataObject]
        """
        choose_sample = []
        count = 0
        for lbl in label_id:
            choose_data = []  # 阈值内数据的index
            unchoose_data = []  # 阈值外数据的index
            for i in machine_samples_list:
                if i.machine_label == lbl:
                    if lower_thres < i.prob < upper_thres:
                        choose_data.append(i)
                    elif i.prob <= lower_thres:
                        unchoose_data.append(i)

            # 阈值内的数据随机抽样batch_num条来标注
            batch_num_part1 = batch_num if batch_num < len(choose_data) else len(choose_data)
            selected_data = random.sample(choose_data, batch_num_part1)

            # 阈值外排序选择预测概率值最低待标样本
            if batch_num_part1 < batch_num:
                batch_num_part2 = batch_num - batch_num_part1
                if unchoose_data:
                    unchoose_data_sort = sorted(unchoose_data, key=lambda d: d.prob)
                    ext_selected_data = unchoose_data_sort[:batch_num_part2]
                    selected_data += ext_selected_data

            count += 1
            for i in selected_data:
                choose_sample.append(i)
        logger.info("choose_sample size：%d, num_classes: %d" % (len(choose_sample), count))
        return choose_sample

    @staticmethod
    def index_by_rule(human_rules, rule_samples, rule_num):
        """
        [DESC]   根据规则抽取样本rule_num个, 返回抽取的下标
        [INPUT]  human_rules:  set, 人工标注结果里的规则集合
                 rule_samples: 机器识别中按规则抽样的数据index集合, rule -> [index]
                 rule_num:     需要按规则抽样个数
        [OUTPUT] rules_index:  [index], 抽取的样本在机器识别结果中的下标
        """
        rules_index = []
        other_rules = set(rule_samples.keys()) - human_rules
        if len(other_rules) == 0:
            if rule_num > len(rule_samples.keys()):
                rule_num = len(rule_samples.keys())
            rules = random.sample(rule_samples.keys(), rule_num)
        else:
            if rule_num > len(other_rules):
                rule_num = len(other_rules)
            rules = random.sample(list(other_rules), rule_num)
        for r in rules:
            rules_index.append(rule_samples[r][0])
        return rules_index

    @staticmethod
    def split_by_threshold(machine_samples_list, lower_thres, upper_thres):
        """
        [DESC]   根据预测的阈值, 从机器识别结果中划分出阈值内和阈值外的index
        [INPUT]  machine_samples_list: 机器识别结果
                 lower_thres:          配置阈值下界
                 upper_thres:          配置阈值上界
        [OUTPUT] trusted_index:        机器识别高置信度的index
                 untrusted_index:      机器识别低置信度的index
        """
        trusted_index = []
        untrusted_index = []
        for s in machine_samples_list:
            if s.prob <= lower_thres or s.prob >= upper_thres:
                trusted_index.append(s)
            else:
                untrusted_index.append(s)
        return trusted_index, untrusted_index

    @staticmethod
    def choose_label_data_by_rule(machine_samples_list, rule_samples, human_rules, batch_num,
                                  lower_thres, upper_thres):
        """
        [DESC]   按规则抽取待标数据，个数为batch_num, 目的覆盖人工标注没覆盖的规则, 覆盖阈值外的样本
        [INPUT]  machine_samples_list: [DataObject], 机器识别结果
                 rule_samples:         {rule->[index]}
                 human_rules:          人工标注结果中已有的规则
                 batch_num:            抽样数
                 lower_thres:          下界
                 upper_thres:          上界
        [OUTPUT] [index]:              选中数据在机器识别结果中的下标
        """
        # 规则，阈值外抽取占比各0.1
        rule_prop = 0.1
        # 规则部分
        rule_num = int(math.ceil(rule_prop * batch_num))
        rule_index = ChooseSamples.index_by_rule(human_rules, rule_samples, rule_num)
        trusted_index, untrusted_index = ChooseSamples.split_by_threshold(machine_samples_list, lower_thres,
                                                                          upper_thres)
        # 阈值外和阈值内部分
        o_index = set(trusted_index) - set(rule_index)
        i_index = set(untrusted_index) - set(rule_index)
        out_thres_index = list(o_index) if rule_num > len(o_index) else \
            random.sample(o_index, rule_num)
        n = batch_num - len(rule_index) - len(out_thres_index)
        n = 0 if n < 0 else n
        in_thres_index = list(i_index) if n > len(i_index) else random.sample(i_index, n)
        # 数量不足随机部分
        other_n = n - len(in_thres_index)
        other_index = []
        if other_n > 0:
            other_index_set = o_index - set(out_thres_index)
            other_index = list(other_index_set) if other_n > len(other_index_set) else \
                random.sample(other_index_set, other_n)
        return rule_index + out_thres_index + in_thres_index + other_index
