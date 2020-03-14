import tensorflow as tf
import pyspark.sql.functions as f
import sparkdl as dl
from pyspark.sql.functions import lit

#root directory
img_dir = "filtered_dataset/"

#currently testing accuracy for 40X magnigication for both malignant and benign 
b_df = dl.readImages(img_dir + "/b40").withColumn("label", lit(1))
m_df = dl.readImages(img_dir + "/m40").withColumn("label", lit(0))


#Splitting the data into training and test in the ratio 80% & 20%
trainb, testb = b_df.randomSplit([80.00, 20.00], seed=42)
trainm, testm = m_df.randomSplit([80.00, 20.00], seed=42)

#combining the dataset benign and malignanent for the training and testing
trainDF = trainb.unionAll(trainm)
testDF = testb.unionAll(testm)

#using Logistic Regression model for training 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
vectorizer = dl.DeepImageFeaturizer(inputCol="image", outputCol="features", modelName='InceptionV3')
logreg = LogisticRegression(maxIter=10,regParam=0.01, elasticNetParam=0.1, labelCol = "label", featuresCol="features")
pipeline = Pipeline(stages=[vectorizer, logreg])

pipeline_model = pipeline.fit(trainDF)


lrModel = pipeline_model
print(lrModel)

lrModel.stages[1].write().overwrite().save('test-1')