# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
import sys

sys.path.append("..")

from labelit import LabelModel

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    file_path = os.path.join(pwd_path, "../extra_data/samples.txt")

    lm = LabelModel(input_file_path=file_path, num_classes=6)
    lm.start_label()
