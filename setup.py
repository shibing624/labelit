# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 
from __future__ import print_function

import sys

from setuptools import setup, find_packages

__version__ = None
exec(open('labelit/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for labelit.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='labelit',
    version=__version__,
    description='label text and image based on active learning.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/labelit',
    license="Apache 2.0",
    zip_safe=False,
    python_requires='>=3.5',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Text Processing :: Linguistic',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='labelit,active learning,label text,label image',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'labelit': 'labelit'},
    package_data={'labelit': ['*.*', '../LICENSE', '../README.*', '../*.txt', 'data/*', ]}
)
