from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr_test = LogisticRegressionModel.load('./test-1')

# Use a featurizer to use trained features from an existing model
featurizer_test = dl.DeepImageFeaturizer(inputCol = "image", outputCol = "features", modelName = "InceptionV3")

# Setup a pipeline
p_lr_test = PipelineModel(stages=[featurizer_test, lr_test])

# Test and evaluate
tested_lr_test = p_lr_test.transform(testDF)
evaluator_lr_test = MulticlassClassificationEvaluator(metricName = "accuracy")
print("Logistic Regression Model: Test set accuracy = " + str(evaluator_lr_test.evaluate(tested_lr_test.select("prediction", "label"))))

tested_lr_test.select("label", "probability", "prediction").show(20, False)