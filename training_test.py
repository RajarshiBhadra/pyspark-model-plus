import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark_model_plus.evaluation import MulticlassLogLossEvaluator
from pyspark_model_plus.training import StratifiedCrossValidator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

import timeit
import time

spark = SparkSession.builder.getOrCreate()


full_iris = pd.read_csv("iris.csv")
train, test = train_test_split(full_iris, stratify=full_iris["Species"], test_size=0.2)
train.append(train[train["Species"] == "setosa"]).append(
    train[train["Species"] == "setosa"]
)
train.to_csv("iris_train.csv", index=False)
test.to_csv("iris_test.csv", index=False)


df_train = spark.read.csv("iris_train.csv", inferSchema=True, header=True)
df_test = spark.read.csv("iris_test.csv", inferSchema=True, header=True)

stages = []
indexer = StringIndexer(inputCol="Species", outputCol="labelIndex")
stages += [indexer]
assembler = VectorAssembler(
    inputCols=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"],
    outputCol="features",
)
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineData = pipeline.fit(df_train)
training_data = pipelineData.transform(df_train)

model = RandomForestClassifier(
    labelCol="labelIndex",
    featuresCol="features",
    probabilityCol="probability",
    predictionCol="prediction",
)
paramGrid = ParamGridBuilder().addGrid(model.numTrees, [250, 252]).build()

parallel = 10

cv = StratifiedCrossValidator(
    labelCol="labelIndex",
    estimator=model,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassLogLossEvaluator(labelCol="labelIndex"),
    numFolds=3,
    stratify_summary=True,
    parallelism=parallel,
)
time_start = time.time()
fitted = cv.fit(training_data)
print("time needed: {}".format(time.time() - time_start))
print("now the measurement starts")
t = timeit.repeat(lambda: cv.fit(training_data), number=2, repeat=3)
print(t)
