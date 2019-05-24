# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

"""
STEP 1: IMPORTING THE DATASET AND PERFORM PREPROCESSING
"""

# IMPORTING DATA SET
Path = (r'C:\\Users\\adong\\Documents\\MachineLearning\\FraudDetection'
        '\\creditcard.csv')

DataCSV = pd.read_csv(Path)

# DROP TIME
DataCSV.drop(['Time'],axis=1,inplace=True)

# X IS THE INDEPENDENT OF THE DATA SET
X = DataCSV.iloc[:, :-1].values

# Y IS THE DEPENDENT VARIABLE OF THE DATA SET
Y = DataCSV.iloc[:, 29].values


"""
STEP 2: SPLITTING THE DATA SET AND THE TRAINING SET
NOTES: THE 0.2 MEANS THAT THERE ARE 20% IN THE TEST SET AND 80% IN THE
TRAINING SET
"""
# NOTE: FROM THE KAGGLE DATASET WE ARE TOLD THAT THE DATA IS
# UNBALANCE THERE IS A BIG DISPARITY BETWEEN FRAUD AND NON FRAUD
overSampler = SMOTE(ratio='auto',random_state=np.random.randint(100),k_neighbors=5,m_neighbors=10,kind='regular')
overSampler_X, overSampler_Y = overSampler.fit_sample(X, Y)

# OVER SAMPLE TRAINING SET
X_train, X_test, Y_train, Y_test = train_test_split(
    overSampler_X, overSampler_Y, test_size=0.2, random_state=np.random.randint(100))

# REGULAR TRAINING SET
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
    X, Y, test_size=0.2, random_state=np.random.randint(100))

# CHECKING NUMBER OF FRUADS IN UNBALANCE DATASET VS BALANCE DATASET
unbalanceFruad = len(Y[Y == 1])
unbalanceNonFraud = len(Y[Y == 0])
balanceFraud = len(overSampler_Y[overSampler_Y == 1])
balanceNonFraud = len(overSampler_Y[overSampler_Y == 0])



"""
STEP 3: TRAIN THE RANDOM FOREST MODEL USING OUR OVER SAMPLE
TRAINING SET
"""

classifier = RandomForestClassifier(
    n_estimators=200, criterion='gini', random_state=0, max_depth=10)
model = classifier.fit(overSampler_X, overSampler_Y)

"""
STEP 4: TEST THE MODEL
"""

Y_prediction = classifier.predict(X_test)


"""
STEP 5: CHECK PERFORMANCE
"""
confusionMatrix = confusion_matrix(Y_test, Y_prediction)
print('Confusion Matrix:')
print(confusionMatrix)
print('Total Test Set Size:')
print(len(X_test))
print('From Test Set ', confusionMatrix[0, 0] +
      confusionMatrix[1, 1], ' data points were successfully predicted')
print('From Test Set ', confusionMatrix[0, 1] +
      confusionMatrix[1, 0], ' data points were not predicted correctly')
print(classification_report(Y_test, Y_prediction))

from sklearn import tree

i_tree = 0
for tree_in_forest in classifier.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file=my_file)
    i_tree = i_tree + 1
    
# DEATURE IMPORTANCE
    
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices],
       color='r', align='center')
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([list(DataCSV.columns)[i] for i in indices], rotation=70)
ax.set_xlim([-1, X.shape[1]])
plt.show()
    
    
