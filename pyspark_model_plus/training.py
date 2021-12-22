from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import Window
from pyspark.sql import functions as f
import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark import keyword_only
from pyspark import inheritable_thread_target
from multiprocessing.pool import ThreadPool


def _parallelFitTasks(est, train, eva, validation, epm, collectSubModel):
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.

    Parameters
    ----------
    est : :py:class:`pyspark.ml.baseEstimator`
        he estimator to be fit.
    train : :py:class:`pyspark.sql.DataFrame`
        DataFrame, training data set, used for fitting.
    eva : :py:class:`pyspark.ml.evaluation.Evaluator`
        used to compute `metric`
    validation : :py:class:`pyspark.sql.DataFrame`
        DataFrame, validation data set, used for evaluation.
    epm : :py:class:`collections.abc.Sequence`
        Sequence of ParamMap, params maps to be used during fitting & evaluation.
    collectSubModel : bool
        Whether to collect sub model.

    Returns
    -------
    tuple
        (int, float, subModel), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    def singleTask():
        index, model = next(modelIter)
        metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric, model if collectSubModel else None

    return [singleTask] * len(epm)


class StratifiedCrossValidator(CrossValidator):
    stratify_summary = Param(
        Params._dummy(),
        "stratify_summary",
        "flag to show stratify summary",
        typeConverter=TypeConverters.toBoolean,
    )

    labelCol = Param(
        Params._dummy(),
        "labelCol",
        "Column containing the labels",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        numFolds=3,
        seed=None,
        parallelism=1,
        collectSubModels=False,
        foldCol="",
        stratify_summary=False,
        labelCol="label",
    ):
        """
        __init__(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                 seed=None, parallelism=1, collectSubModels=False, foldCol="",\
                 stratify_summary=False, labelCol="bool"):
        """
        super(CrossValidator, self).__init__()
        self._setDefault(parallelism=1, stratify_summary=False, labelCol="label")
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(
        self,
        *,
        estimator=None,
        estimatorParamMaps=None,
        evaluator=None,
        numFolds=3,
        seed=None,
        parallelism=1,
        collectSubModels=False,
        foldCol="",
        stratify_summary=False,
        labelCol="label",
    ):
        """
        setParams(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                  seed=None, parallelism=1, collectSubModels=False, foldCol="",\
                  stratify_summary=False, labelCol="bool"):
        Sets params for cross validator.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setStratifySummary(self, value):
        """
        Sets the value of stratify_summary flag.
        """
        return self._set(stratify_summary=value)

    def getStratifySummary(self):
        return self.getOrDefault(self.stratify_summary)

    def setLabelCol(self, value):
        """
        Set the value for the variable labelCol
        """
        return self._set(labelCol=value)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)

    def _stratify_data(self, dataset):
        nfolds = self.getOrDefault(self.numFolds)
        labelCol = self.getOrDefault(self.labelCol)
        stratify_summary = self.getOrDefault(self.stratify_summary)
        df = dataset.withColumn("id", monotonically_increasing_id())
        windowval = (
            Window.partitionBy(labelCol)
            .orderBy("id")
            .rangeBetween(Window.unboundedPreceding, 0)
        )
        stratified_data = df.withColumn("cum_sum", f.sum(f.lit(1)).over(windowval))
        stratified_data = stratified_data.withColumn(
            "bucket_fold", f.col("cum_sum") % nfolds
        )

        if stratify_summary:
            stratify_summary = (
                stratified_data.withColumn(
                    "bucket_fold", f.concat(f.lit("fold_"), f.col("bucket_fold") + 1)
                )
                .groupby(labelCol)
                .pivot("bucket_fold")
                .agg(f.count("id"))
            )
            print(stratify_summary.toPandas())

        stratified_data = stratified_data.drop(*["id", "cum_sum"])

        return stratified_data

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        metrics = [0.0] * numModels

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()

        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        stratified_data = self._stratify_data(dataset)

        for i in range(nFolds):

            train = stratified_data.filter(stratified_data["bucket_fold"] != i)
            validation = stratified_data.filter(stratified_data["bucket_fold"] == i)

            tasks = map(
                inheritable_thread_target,
                _parallelFitTasks(
                    est, train, eva, validation, epm, collectSubModelsParam
                ),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += metric / nFolds
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels))
