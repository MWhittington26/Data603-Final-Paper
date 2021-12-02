import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Create H2O instance and read in the dataset
h2o.init()
data = h2o.import_file('healthcare-dataset-stroke-data.csv')
# Convert the categorical variables
data["gender"]=data["gender"].asfactor()
data["hypertension"]=data["hypertension"].asfactor()
data["heart_disease"]=data["heart_disease"].asfactor()
data["ever_married"]=data["ever_married"].asfactor()
data["work_type"]=data["work_type"].asfactor()
data["Residence_type"]=data["Residence_type"].asfactor()
data["smoking_status"]=data["smoking_status"].asfactor()
data["stroke"]=data["stroke"].asfactor()
# Use the proper encoding, identify the training columns, response column
encoding = 'one_hot_explicit'
training_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
response_column = 'stroke'
# Split into train and test
train, test = data.split_frame(ratios=[0.8])
# Build the Random Forest Model
model = H2ORandomForestEstimator(categorical_encoding=encoding)
model.train(x=training_columns, y=response_column, training_frame=train)
# Print out the performance, prediction, and accuracy of the model
performance = model.model_performance(test_data=test)
print(performance)
predict = model.predict(test_data=test)
print(predict)
accuracy = model.accuracy()
print(accuracy)