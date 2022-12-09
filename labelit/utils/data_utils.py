# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import pickle
import jieba
from codecs import open
from collections import defaultdict
from loguru import logger


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            item = item if not lower else item.lower()
            dic[item] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    return result


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    logger.info("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write(word + '\n')
            else:
                f.write(word)
    logger.info("- write to {} done. {} tokens".format(filename, len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise IOError(filename)
    return d


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("save to {} ok.".format(pkl_path))


def get_char_segment_data(contents, word_sep=' ', pos_sep='/'):
    """
    获取分字数据
    """
    data = []
    for content in contents:
        temp = ''
        for word in content.split(word_sep):
            if pos_sep in word:
                temp += word.split(pos_sep)[0]
            else:
                temp += word.strip()
        temp = ' '.join(list(temp))
        data.append(temp)
    return data


def load_list(path):
    """
    读取列表
    """
    if not os.path.exists(path):
        return []
    return [word for word in open(path, 'r', encoding='utf-8').read().split()]


def save_predict_result(pred_labels, ture_labels=None, pred_save_path=None, data_set=None):
    if pred_save_path:
        with open(pred_save_path, 'w', encoding='utf-8') as f:
            for i in range(len(pred_labels)):
                if ture_labels and len(ture_labels) > 0:
                    assert len(ture_labels) == len(pred_labels)
                    if data_set:
                        f.write(ture_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(ture_labels[i] + '\n')
                else:
                    if data_set:
                        f.write(pred_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(pred_labels[i] + '\n')
        logger.info("pred_save_path: {}".format(pred_save_path))


def data_reader(path, col_sep='\t', stopwords_path='', segment_type='word', lower=False):
    """
    读取数据，并分词获取词列表
    """
    contents = []
    labels = []
    seg_contents = []
    word_list = []
    stopwords = read_stopwords(stopwords_path)
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if col_sep in line:
                    index = line.index(col_sep)
                    labels.append(line[:index].strip())
                    content = line[index + 1:].strip()
                else:
                    content = line
                content = content.lower() if lower else content
                contents.append(content)
                words = seg_sentence(content, stopwords, segment_type)
                word_list.extend(words)
                seg_contents.append(' '.join(words))
    return contents, seg_contents, labels, word_list


def read_stopwords(path):
    """
    读取停用词
    """
    lines = set()
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                lines.add(line)
    return lines


def seg_sentence(sentence, stopwords=None, segment_type='word'):
    """
    预处理（切词，去除停用词）
    :param sentence:
    :param stopwords:
    :param segment_type: char/word
    :return:
    """
    stopwords = stopwords if stopwords else set()
    sentence = sentence.strip()
    if segment_type == 'word':
        words = jieba.cut(sentence)
    else:
        words = list(sentence)
    words = [word for word in words if word not in stopwords]
    return words
