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

    lm = LabelModel(input_file_path=file_path, model_type='logistic', num_classes=6)
    noise_samples, label_issues_df = lm.find_noise()
    print(f"noise_samples len: {len(noise_samples)}")
    label_issues_df.to_csv("label_issues.csv", index=False)
    lm.train_with_no_noise()
