# pyspark-model-plus
[![PyPI version](https://img.shields.io/pypi/v/pyspark-model-plus.svg)](https://img.shields.io/pypi/v/pyspark-model-plus)

This package has been written keeping in mind some functions that we commonly use in scikit-learn but are not currently available in 
spark machine learning library. Capabilities the package is adding are

* Multi Class LogLoss Evaluator
* Stratified Cross Validation
* Impute multiple columns by column mean (faster)

## About the functions

**MulticlassLogLossEvaluator**

[Spark documentaion](https://spark.apache.org/docs/1.6.0/mllib-evaluation-metrics.html) mentions currently there is no existing function available in default spark mllib to perform logloss evaluation for categorical variables. The corresponding function that enables us to perform this in scikit-learn is [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).This function is an attempt to add that functionality so that it can be used with standard ML pipelines. The core idea for the algorthm has been made on the basis of this [post](http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209).

**StratifiedCrossValidator**  

Stratified sampling is important for hyper parameter tuning during CV fold training as it enables us keep the final tuning parameters robust against sampling bias speciaally whe the data is unabalanced. [Spark documentation](https://spark.apache.org/docs/latest/ml-tuning.html#cross-validation) indicates that we currently cannot do that. As a result many approaches has been proposed to include this in pyspark. For example [spark_stratifier](https://github.com/interviewstreet/spark-stratifier) implements this functionality but with two major drawbacks

* The algorithm is dependent on joins
* It only works for binary classification problems(as of now)

This function tries to address both the issues by making the function independant of joins and also making the approach general such that startified cross validation can be done on multiclass classification problems as well

**CustomMeanImputer**  

[Spark documentation](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=impute#pyspark.ml.feature.Imputer) shows that a imputer class exists. However for large data sets using for loop in this function makes it slow. This function tried to address that usse by tryinjg to do impute by mean simulateneously using agg and python distionary

## Requirements

The package currently requires only [`numpy`](https://github.com/numpy/numpy) and [`pyspark`](https://github.com/apache/spark/tree/master/python/pyspark).

## Installation

