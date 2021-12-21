import pandas as pd
import time
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession

from pyspark_model_plus.evaluation import MulticlassLogLossEvaluator
from pyspark_model_plus.training import StratifiedCrossValidator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

import perfplot

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
print(training_data.head(1))
# print(training_data.explain(True))

model = RandomForestClassifier(
    labelCol="labelIndex",
    featuresCol="features",
    probabilityCol="probability",
    predictionCol="prediction",
)
paramGrid = ParamGridBuilder().addGrid(model.numTrees, [250, 252]).build()

cv_dict = {}

for parallel in range(1, 11):
    cv_dict[parallel] = StratifiedCrossValidator(
        labelCol="labelIndex",
        estimator=model,
        estimatorParamMaps=paramGrid,
        evaluator=MulticlassLogLossEvaluator(labelCol="labelIndex"),
        numFolds=3,
        stratify_summary=True,
        parallelism=parallel,
    )
print("works")
out = perfplot.bench(
    setup=lambda n: training_data,
    kernels=[lambda df: cv.fit(df) for cv in cv_dict.values()],
    labels=[str(cv) for cv in cv_dict.keys()],
    n_range=list(range(2)),  # how often for one method
    xlabel="#loop",
)
out.show()
out.save("perf.png", transparent=True, bbox_inches="tight")
## stratifiedCV = StratifiedCrossValidator(cv)
# start_time = time.time()
# cvModel = cv.fit(training_data)
# print("parallel = {}, time = {}".format(parallel, time.time() - start_time))

# test_data = pipelineData.transform(df_test)
# predictions = cvModel.transform(test_data)
# evaluator = MulticlassLogLossEvaluator(labelCol="labelIndex")
# accuracy = evaluator.evaluate(predictions)
# print(accuracy)
