from pyspark.ml.evaluation import Evaluator
from pyspark.sql import functions as f
import numpy as np


class MulticlassLogLossEvaluator(Evaluator):
    def __init__(self,labelCol="label"):
        """
        __init__(self, predictionCol="prediction", labelCol="label", \
                 metricName="f1Measure", metricLabel=0.0)
        """
        super(MulticlassLogLossEvaluator, self).__init__()
        self.labelCol = labelCol
        #kwargs = self._input_kwargs
        #self._set(**kwargs)
    def _evaluate(self, dataset):
        # Exploding the probability vector type into mutliple columns
        # Change the row.value_segment > to the row.segment_name

        def calc_log(row):
  
            predict_proba = row.probability.toArray().tolist()
            label_index = row['labelIndex']

            eps=1e-15 
            predict_proba_clipped = np.clip(predict_proba, eps, 1 - eps)

            #Normalized
            predict_proba_clipped /= predict_proba_clipped.sum()

            label_index = int(row['labelIndex'])
            selected_proba = predict_proba_clipped[label_index]

            log_selected_proba = np.log(selected_proba).tolist()

            return tuple([log_selected_proba])

        predicted = dataset.rdd.map(calc_log).toDF()
        reduce_step = predicted.agg(f.sum("_1"), f.count("_1")).collect()
        log_sum = reduce_step[0][0]
        count =   reduce_step[0][1]
        LogLoss = -1.0 / count * log_sum
        
        if LogLoss is None:
            LogLoss = .5
        return LogLoss

    def isLargerBetter(self):
        return False

    def getMetricName(self):
        return "Multinomial Log Loss"