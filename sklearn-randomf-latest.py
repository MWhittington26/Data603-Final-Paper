from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Read in the dataset using pandas
df = pd.read_csv('C:/Users/whitt/OneDrive/Documents/DATA 603/Final Project/healthcare-dataset-stroke-data.csv')

# Find any missing values
isnull = df.isnull().sum()
print(isnull)

# BMI mean calc - this will act in place of the null values
bmimean = df['bmi'].mean()
print(bmimean)

# Fill the bmi null values with the mean
df = df.fillna(bmimean)
isnull = df.isnull().sum()
print(isnull)

# Describe data
dtypes = df.dtypes
print(dtypes)

# Drop id since it is not relevant in the prediction
df.drop('id', axis = 1,inplace = True)

# Conversion of categorical data to numbers using dummy variables
gender_dum = pd.get_dummies(df['gender'])
residence_type_dum = pd.get_dummies(df['Residence_type'])
smoking_status_dum = pd.get_dummies(df['smoking_status'])
work_type_dum = pd.get_dummies(df['work_type'])
ever_marries_dum = pd.get_dummies(df['ever_married'])

df = pd.concat([df, gender_dum, residence_type_dum, smoking_status_dum, work_type_dum],axis='columns')
df = df.drop(columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
print(df)
# Identify the column that will be analyzed 'stroke'
X = df.drop('stroke',axis=1)
y = df['stroke']
print(y)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
traininglength = len(X_train), len(X_test)
print(traininglength)
# Develop RF Classification model
rfc = RandomForestClassifier()
# Fit, evaluate and get the accuracy of the model
rfc.fit(X_train,y_train)
rfcclassifierscore = rfc.score(X_test, y_test)
print('Accuracy Score:',rfcclassifierscore*100)
