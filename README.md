# labelit
label text and image based on active learning.

# Active Learning Playground

## Introduction

This is a python module for experimenting with different active learning
algorithms. There are a few key components to running active learning
experiments:

*   Main experiment script is
    [`run_experiment.py`](run_experiment.py)
    with many flags for different run options.

*   Supported datasets can be downloaded to a specified directory by running
    [`utils/create_data.py`](utils/create_data.py).

*   Supported active learning methods are in
    [`sampling_methods`](sampling_methods/).

Below I will go into each component in more detail.

DISCLAIMER: This is not an official Google product.

## Setup
The dependencies are in [`requirements.txt`](requirements.txt).  Please make sure these packages are
installed before running experiments.  If GPU capable `tensorflow` is desired, please follow
instructions [here](https://www.tensorflow.org/install/).

It is highly suggested that you install all dependencies into a separate `virtualenv` for
easy package management.

## Getting benchmark datasets

By default the datasets are saved to `/tmp/data`. You can specify another directory via the
`--save_dir` flag.

Redownloading all the datasets will be very time consuming so please be patient.
You can specify a subset of the data to download by passing in a comma separated
string of datasets via the `--datasets` flag.

## Running experiments

There are a few key flags for
[`run_experiment.py`](run_experiment.py):

*   `dataset`: name of the dataset, must match the save name used in
    `create_data.py`. Must also exist in the data_dir.

*   `sampling_method`: active learning method to use. Must be specified in
    [`sampling_methods/constants.py`](sampling_methods/constants.py).

*   `warmstart_size`: initial batch of uniformly sampled examples to use as seed
    data. Float indicates percentage of total training data and integer
    indicates raw size.

*   `batch_size`: number of datapoints to request in each batch. Float indicates
    percentage of total training data and integer indicates raw size.

*   `score_method`: model to use to evaluate the performance of the sampling
    method. Must be in `get_model` method of
    [`utils/utils.py`](utils/utils.py).

*   `data_dir`: directory with saved datasets.

*   `save_dir`: directory to save results.

This is just a subset of all the flags. There are also options for
preprocessing, introducing labeling noise, dataset subsampling, and using a
different model to select than to score/evaluate.

## Available active learning methods

All named active learning methods are in
[`sampling_methods/constants.py`](sampling_methods/constants.py).

You can also specify a mixture of active learning methods by following the
pattern of `[sampling_method]-[mixture_weight]` separated by dashes; i.e.
`mixture_of_samplers-margin-0.33-informative_diverse-0.33-uniform-0.34`.

Some supported sampling methods include:

*   Uniform: samples are selected via uniform sampling.

*   Margin: uncertainty based sampling method.

*   Informative and diverse: margin and cluster based sampling method.

*   k-center greedy: representative strategy that greedily forms a batch of
    points to minimize maximum distance from a labeled point.

*   Graph density: representative strategy that selects points in dense regions
    of pool.

*   Exp3 bandit: meta-active learning method that tries to learns optimal
    sampling method using a popular multi-armed bandit algorithm.

### Adding new active learning methods

Implement either a base sampler that inherits from
[`SamplingMethod`](sampling_methods/sampling_def.py)
or a meta-sampler that calls base samplers which inherits from
[`WrapperSamplingMethod`](sampling_methods/wrapper_sampler_def.py).

The only method that must be implemented by any sampler is `select_batch_`,
which can have arbitrary named arguments. The only restriction is that the name
for the same input must be consistent across all the samplers (i.e. the indices
for already selected examples all have the same name across samplers). Adding a
new named argument that hasn't been used in other sampling methods will require
feeding that into the `select_batch` call in
[`run_experiment.py`](run_experiment.py).

After implementing your sampler, be sure to add it to
[`constants.py`](sampling_methods/constants.py)
so that it can be called from
[`run_experiment.py`](run_experiment.py).

## Available models

All available models are in the `get_model` method of
[`utils/utils.py`](utils/utils.py).

Supported methods:

*   Linear SVM: scikit method with grid search wrapper for regularization
    parameter.

*   Kernel SVM: scikit method with grid search wrapper for regularization
    parameter.

*   Logistc Regression: scikit method with grid search wrapper for
    regularization parameter.

*   Small CNN: 4 layer CNN optimized using rmsprop implemented in Keras with
    tensorflow backend.

*   Kernel Least Squares Classification: block gradient descient solver that can
    use multiple cores so is often faster than scikit Kernel SVM.

### Adding new models

New models must follow the scikit learn api and implement the following methods

*   `fit(X, y[, sample_weight])`: fit the model to the input features and
    target.

*   `predict(X)`: predict the value of the input features.

*   `score(X, y)`: returns target metric given test features and test targets.

*   `decision_function(X)` (optional): return class probabilities, distance to
    decision boundaries, or other metric that can be used by margin sampler as a
    measure of uncertainty.

See
[`small_cnn.py`](utils/small_cnn.py)
for an example.

After implementing your new model, be sure to add it to `get_model` method of
[`utils/utils.py`](utils/utils.py).

Currently models must be added on a one-off basis and not all scikit-learn
classifiers are supported due to the need for user input on whether and how to
tune the hyperparameters of the model. However, it is very easy to add a
scikit-learn model with hyperparameter search wrapped around as a supported
model.

## Collecting results and charting

The
[`utils/chart_data.py`](utils/chart_data.py)
script handles processing of data and charting for a specified dataset and
source directory.


# 主动学习
在某些情况下，没有类标签的数据相当丰富而有类标签的数据相当稀少，并且人工对数据进行标记的成本又相当高昂。在这种情况下，我们可以让学习算法主动地提出要对哪些数据进行标注，之后我们要将这些数据送到专家那里进行标注，再将这些数据加入到训练样本集中对算法进行训练。这一过程叫做主动学习。

主动学习方法一般可以分为两部分： 学习引擎和选择引擎。学习引擎维护一个基准分类器，并使用监督学习算法学习已标注的样例，进而提高该分类器的性能，而选择引擎通过样例选择算法选择一个未标注的样例并将其交由人类专家进行标注，再将标注后的样例加入到已标注样例集。学习引擎和选择引擎交替工作，经过多次循环，基准分类器的性能逐渐提高，当满足预设条件时，过程终止。

# 样例选择算法
根据获得未标注样例的方式，可以将主动学习分为两种类型：基于流的和基于池的。

- 基于池(pool-based)的主动学习中则维护一个未标注样例的集合，由选择引擎在该集合中选择当前要标注的样例。
- 基于流(stream-based)的主动学习中，未标记的样例按先后顺序逐个提交给选择引擎，由选择引擎决定是否标注当前提交的样例，如果不标注，则将其丢弃。由于基于流的算法不能对未标注样例逐一比较，需要对样例的相应评价指标设定阈值，当提交给选择引擎的样例评价指标超过阈值，则进行标注，但这种方法需要针对不同的任务进行调整，所以难以作为一种成熟的方法投入使用。此处不再介绍。

## 基于池的样例选择算法

1. 基于不确定度缩减的方法

这类方法选择那些当前基准分类器最不能确定其分类的样例进行标注。这类方法以信息熵作为衡量样例所含信息量大小的度量，而信息熵最大的样例正是当前分类器最不能确定其分类的样例。从几何角度看，这种方法优先选择靠近分类边界的样例。

2. 基于版本缩减的方法

这类方法选择那些训练后能够最大程度缩减版本空间的样例进行标注。在二值分类问题中，这类方法选择的样例总是差不多平分版本空间。

代表：QBC算法

QBC算法从版本空间中随机选择若干假设构成一个委员会，然后选择委员会中的假设预测分歧最大的样例进行标注。为了优化委员会的构成，可以采用Bagging,AdaBoost等分类器集成算法从版本空间中产生委员会。

3. 基于泛化误差缩减的方法

这类方法试图选择那些能够使未来泛化误差最大程度减小的样例。其一般过程为：首先选择一个损失函数用于估计未来错误率，然后将未标注样例集中的每一个样例都分别估计其能给基准分类器带来的误差缩减，选择估计值最大的那个样例进行标注。

这类方法直接针对分类器性能的最终评价指标，但是计算量较大，同时损失函数的精度对性能影响较大。

4. 其它方法

- COMB算法：组合三种不同的学习器，迅速切换到当前性能最好的学习器从而使选择样例尽可能高效。

- 多视图主动学习：用于学习问题为多视图学习的情况，选择那些使不同视图的预测分类不一致的样例进行学习。这种方法对于处理高维的主动学习问题非常有效。

- 预聚类主动学习：预先运行聚类算法预处理，选择样例时优先选择最靠近分类边界的样例和最能代表聚类的样例（即聚类中心）。

# 应用
## 文档分类和信息提取
以贝叶斯方法位基准分类器，使用基于不确定度缩减的样例选择算法进行文本分类。

将EM算法同基于QBC方法的主动学习集合。EM算法能够有效的利用未标注样例中的信息提高基准分类器的分类正确率。而QBC方法能够迅速缩减版本空间。

## 图像检索
利用SVM作为基准分类器的主动学习算法来处理图像检索。该算法采用最近边界方法作为样例选择算法，同时将图像的颜色、纹理等提取出来作为部分特征进行学习。

## 入侵检测
由于入侵检测系统较多地依赖专家知识和有效的数据集，所以可以采用主动学习算法降低这种依赖性。


# Usage
1. python3 label.py


