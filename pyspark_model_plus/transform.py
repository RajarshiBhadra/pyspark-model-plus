from pyspark.ml import Estimator
import re


class CustomMeanImputer(Estimator):
    """
    This class replaces the missing values of non-categorical variables with 
    their mean value calculated using aggregate

        note: CustomImputer class was created as there was no inbuilt class for 
        imputing missing values in Pyspark 2.1.0.
         It was continued after upgrade to Pyspark 2.2.0 as the new Imputer 
         class did operations column by column which made the process slow

    """

    def __init__(self, cols_to_impute, value=None):
        super(CustomMeanImputer, self).__init__()
        self.value = value
        self.cols_to_impute = cols_to_impute

    def setParams(self, inputCol, cols_to_impute, value=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def getImputeValue(self):
        print(self.inputCol, self.value)

    def _fit(self, dataset):
        num_cols = self.cols_to_impute
        self.value = dict()
        dict_custom = {key: "avg" for key in num_cols}
        mean_dict = dataset.agg(dict_custom).first().asDict()
        self.value = {
            re.findall(r"avg\((.*)\)", key)[0]: round(float(val), 4)
            for key, val in mean_dict.items()
        }
        return CustomMeanImputer(value=self.value, cols_to_impute=self.cols_to_impute)

    def _transform(self, dataset):
        dataset = dataset.na.fill(self.value)
        return dataset
