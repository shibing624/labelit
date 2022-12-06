# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from codecs import open
from time import time
from loguru import logger


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def seg_data(in_file, out_file, col_sep='\t', stop_words_path='', segment_type='word'):
    """
    预处理（切词，去除停用词）
    :param in_file:
    :param out_file:
    :param col_sep:
    :param stop_words_path:
    :param segment_type: char/word
    :return:
    """
    stopwords = read_stopwords(stop_words_path)
    with open(in_file, 'r', encoding='utf-8') as f1, open(out_file, 'w', encoding='utf-8') as f2:
        count = 0
        for line in f1:
            line = line.rstrip()
            parts = line.split(col_sep)
            if len(parts) < 2:
                continue
            label = parts[0].strip()
            data = ' '.join(parts[1:])
            if segment_type == 'word':
                import jieba
                seg_list = jieba.lcut(data)
                seg_words = []
                for i in seg_list:
                    if i in stopwords:
                        continue
                    seg_words.append(i)
                seg_line = ' '.join(seg_words)
            else:
                seg_line = ' '.join(list(data))
            if count % 10000 == 0:
                logger.debug('count:%d' % count)
                logger.debug(line)
                logger.debug('=' * 20)
                logger.debug(seg_line)
            count += 1
            f2.write('%s\t%s\n' % (label, seg_line))
        logger.info('%s to %s, size: %d' % (in_file, out_file, count))


if __name__ == '__main__':
    import sys

    sys.path.append('.')
    from labelit import config

    start_time = time()
    seg_data(config.input_file_path, config.seg_input_file_path, col_sep=config.col_sep,
             stop_words_path=config.stop_words_path, segment_type='word')
    print("spend time: %s s" % (time() - start_time))
