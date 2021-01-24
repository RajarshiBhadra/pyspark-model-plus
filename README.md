# pyspark-model-plus
[![PyPI version](https://img.shields.io/pypi/v/pyspark-model-plus.svg)](https://img.shields.io/pypi/v/pyspark-model-plus)

This package has been written keeping in mind some functions that we commonly use in scikit-learn but are not currently available in 
spark machine learning library. Capabilities the package is currently adding are

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
```
$ pip install https://test.pypi.org/simple/ pyspark-model-plus-rbhadra90==0.0.12
```
## How to use

Here is an example on how to use the function using the [iris data](https://archive.ics.uci.edu/ml/datasets/iris).
Let us first try to split the data using scikit learn's train test split functionality

```py
import pandas as pd
from sklearn.model_selection import train_test_split

full_iris = pd.read_csv("iris.csv")
train,test = train_test_split(full_iris,stratify = full_iris["Species"],test_size = .2)
train.append(train[train["Species"] == "setosa"]).\
      append(train[train["Species"] == "setosa"]).to_csv("iris_train.csv", index = False)
test.to_csv("iris_test.csv", index = False)
```

**Importing Packages**

```py
from pyspark_model_plus.evaluation import MulticlassLogLossEvaluator
from pyspark_model_plus.training import StratifiedCrossValidator
from pyspark_model_plus.transform import CustomMeanImputer
```


**Importing to pyspark**

```py
df_train = spark.read.csv("iris_train.csv", inferSchema=True, header=True)
df_test = spark.read.csv("iris_test.csv", inferSchema=True, header=True)
```

**Creating pipeline to prepare training data**

```py
stages = []
indexer = StringIndexer(inputCol="Species", outputCol="labelIndex")
stages += [indexer]
imputer = CustomMeanImputer(cols_to_impute = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
stages += [imputer]
assembler = VectorAssembler(
    inputCols=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"],
    outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineData = pipeline.fit(df_train)
training_data = pipelineData.transform(df_train)

```

**Training RandomForest on unbalanced data using Stratified Crossvalidation**

```py
model = RandomForestClassifier(labelCol="labelIndex",
                               featuresCol="features",
                               probabilityCol="probability",
                               predictionCol="prediction")
paramGrid = (ParamGridBuilder().addGrid(model.numTrees, [250, 300]).build())
cv = StratifiedCrossValidator(
    labelCol = "labelIndex",
    estimator=model,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassLogLossEvaluator(labelCol="labelIndex"),
    numFolds=3,
    stratify_summary = True
)

# stratifiedCV = StratifiedCrossValidator(cv)
cvModel = cv.fit(training_data)
```
As the training progresses you will see the progress being printed

```py
Initiating Training for fold 1
Initiating Training for fold 2
Initiating Training for fold 3
Out[19]: [0.06820200113875617, 0.06960025759185091]
```

Additionally if you want to see how the startified crossvalidator is splitting the data for each fold, you can run with the `stratify_summary=True` and see the report as below

```py
+----------+------+------+------+
|labelIndex|fold_1|fold_2|fold_3|
+----------+------+------+------+
|       0.0|    40|    40|    40|
|       1.0|    13|    14|    13|
|       2.0|    13|    14|    13|
+----------+------+------+------+
```
# Evaluate and compare with scikit learn

### With MulticlassLogLossEvaluator

```py
test_data = pipelineData.transform(df_test)
predictions = cvModel.transform(test_data)
evaluator = MulticlassLogLossEvaluator(labelCol="labelIndex")
accuracy = evaluator.evaluate(predictions)
accuracy

Out[23]: 0.07676493894621013
```

### With scikit-learn

```py
from sklearn.metrics import log_loss
predictions_pandas = predictions.toPandas()
log_loss(predictions_pandas["labelIndex"].tolist(),predictions_pandas["probability"].tolist())

Out[24]: 0.0767649389462101
```

# Contributing

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/RajarshiBhadra/pyspark-model-plus/issues)

If you want to write some code and contribute to this project, go ahead and start a pull request. I hope this tool is useful for the community and I would love to hear about how this helps solve your problems!
