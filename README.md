# Heart Disease Prediction
This Python script is designed to predict the presence of heart disease based on various health-related features. The script utilizes a logistic regression model and includes data collection, processing, model training, evaluation, and a simple predictive system.

## Getting Started
### Prerequisites
Python (>=3.6)
### Required libraries: 
NumPy, Pandas, Seaborn, scikit-learn
### Installation
Install the necessary libraries using the following command:

pip install numpy pandas seaborn scikit-learn

## Usage
Clone the repository:
Place your dataset file (heart_disease_data.csv) in the same directory as the script.

### Run the script:
python heart_disease_prediction.py
Code Structure
#### Import Dependencies

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

#### Data Collection and Processing

df = pd.read_csv("heart_disease_data.csv")

#### Check for Null Values

df.isna().sum()

#### Data Exploration


df.describe()

df.info()

#### Data Visualization


corelation = df.corr()

sns.heatmap(corelation, cbar=True, annot=True, cmap="Blues", fmt=".1f", square=True)

#### Split Features and Target


X = df.drop("target", axis=1)

Y = df.target

#### Splitting Data into Training and Test Sets


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=2)

#### Train Logistic Regression Model


logr = LogisticRegression(max_iter=1000)

logr.fit(X_train, Y_train)

#### Evaluate Model on Training Data


X_train_predicted = logr.predict(X_train)

accuracy_score(Y_train, X_train_predicted)

#### Evaluate Model on Test Data


X_test_predicted = logr.predict(X_test)

accuracy_score(Y_test, X_test_predicted)

#### Build a Predictive System


input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

input_data_np = np.asarray(input_data)

input_data_np_reshaped = input_data_np.reshape(1, -1)

prediction = logr.predict(input_data_np_reshaped)

print(prediction)

if prediction[0] == 0:

    print('The Person does not have a Heart Disease')
    
else:
    print('The Person has Heart Disease')
    
## Contact
For any enquiries please contact me at :
      
      mitanshubaranwal70232@gmail.com
      
