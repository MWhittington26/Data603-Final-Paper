from pyspark.sql import SparkSession
from pyspark.ml.feature import (VectorAssembler,OneHotEncoder,StringIndexer)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create SparkSession and read in the dataset
spark = SparkSession.builder.appName('stroke').getOrCreate()
df = spark.read.csv('healthcare-dataset-stroke-data.csv', header = True, inferSchema = True)

changed_df = df.withColumn("bmi", col("bmi").cast("double"))
changed_df.printSchema()
changed_df.groupBy('stroke').count().show()

# Fill in missing values with bmi mean
from pyspark.sql.functions import mean
mean = changed_df.select(mean(changed_df['bmi'])).collect()
mean_bmi = mean[0][0]
df_new = changed_df.na.fill(mean_bmi,['bmi'])
# Index and encode the categorical features
gender_indexer = StringIndexer(inputCol='gender',outputCol='genderIndex')
gender_encoder = OneHotEncoder(inputCol='genderIndex',outputCol='genderVec')

ever_married_indexer = StringIndexer(inputCol='ever_married',outputCol='ever_marriedIndex')
ever_married_encoder = OneHotEncoder(inputCol='ever_marriedIndex',outputCol='ever_marriedVec')

work_type_indexer = StringIndexer(inputCol='work_type',outputCol='work_typeIndex')
work_type_encoder = OneHotEncoder(inputCol='work_typeIndex',outputCol='work_typeVec')

Residence_type_indexer = StringIndexer(inputCol='Residence_type',outputCol='Residence_typeIndex')
Residence_type_encoder = OneHotEncoder(inputCol='Residence_typeIndex',outputCol='Residence_typeVec')

smoking_status_indexer = StringIndexer(inputCol='smoking_status',outputCol='smoking_statusIndex')
smoking_status_encoder = OneHotEncoder(inputCol='smoking_statusIndex',outputCol='smoking_statusVec')
# Create features column
assembler = VectorAssembler(inputCols=['genderVec',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedVec',
 'work_typeVec',
 'Residence_typeVec',
 'avg_glucose_level',
 'bmi',
 'smoking_statusVec'],outputCol='features')
# Develop RF Classification model
rfc = RandomForestClassifier(labelCol='stroke', featuresCol='features')
# Create the pipeline
pipeline = Pipeline(stages=[gender_indexer, ever_married_indexer, work_type_indexer, Residence_type_indexer, smoking_status_indexer, gender_encoder, ever_married_encoder, work_type_encoder, Residence_type_encoder, smoking_status_encoder, assembler, rfc])
# Split, Fit, and Transform the data
train_data,test_data = df_new.randomSplit([0.8,0.2])
model = pipeline.fit(train_data)
rfc_predictions = model.transform(test_data)
# Evaluate and get the accuracy of the model
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
print('Random Forest algorithm had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))