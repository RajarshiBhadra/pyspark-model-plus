from pyspark.ml.tuning import CrossValidator
from pyspark.sql import Window
from pyspark.sql import functions as f
import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel


class StratifiedCrossValidatorOld(CrossValidator):
    def __init__(
        self,
        estimator,
        estimatorParamMaps,
        evaluator,
        numFolds,
        stratify_summary=False,
        labelCol="label",
    ):
        super(CrossValidator, self).__init__()
        self.labelCol = labelCol
        self.estimator = estimator
        self.estimatorParamMaps = estimatorParamMaps
        self.evaluator = evaluator
        self.numFolds = numFolds
        self.stratify_summary = stratify_summary

    def stratify_data(self, dataset):
        nfolds = self.numFolds
        df = dataset.withColumn("id", monotonically_increasing_id())
        windowval = (
            Window.partitionBy(self.labelCol)
            .orderBy("id")
            .rangeBetween(Window.unboundedPreceding, 0)
        )
        stratified_data = df.withColumn("cum_sum", f.sum(f.lit(1)).over(windowval))
        stratified_data = stratified_data.withColumn(
            "bucket_fold", f.col("cum_sum") % nfolds
        )

        if self.stratify_summary:
            self.stratify_summary = (
                stratified_data.withColumn(
                    "bucket_fold", f.concat(f.lit("fold_"), f.col("bucket_fold") + 1)
                )
                .groupby(self.labelCol)
                .pivot("bucket_fold")
                .agg(f.count("id"))
            )
        else:
            self.stratify_summary = (
                "To create summary rerun with stratify_summary = True"
            )

        stratified_data = stratified_data.drop(*["id", "cum_sum"])

        return stratified_data

    def _fit(self, dataset):
        est = self.estimator
        epm = self.estimatorParamMaps
        numModels = len(epm)
        eva = self.evaluator
        metricName = eva.getMetricName()
        nFolds = self.numFolds
        metrics = [0.0] * numModels
        stratified_data = self.stratify_data(dataset)

        for i in range(nFolds):

            train = stratified_data.filter(stratified_data["bucket_fold"] != i)
            validation = stratified_data.filter(stratified_data["bucket_fold"] == i)

            models = est.fit(train, epm)

            for j in range(numModels):
                model = models[j]
                metric = eva.evaluate(model.transform(validation, epm[j]))
                metrics[j] += metric / nFolds

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)

        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics))
