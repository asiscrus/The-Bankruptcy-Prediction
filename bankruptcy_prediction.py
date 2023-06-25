# Python codes: Statistical tests, Neural Net solution, and CatBoost Classifier
# In this first step, the necessary packages are imported in Jupyter Notebook 
# In the dataset, inactive firms are defined as 1 instead of 0. 
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# load data
data=pd.read_csv('_dataa.csv', sep=';')
data = data.dropna()

# In this part, the count and rates of Class 0 and Class 1 are calculated.
count_1 = data['active_inactive'].sum()
count_0 = len(data['active_inactive']) - count_1

print('The count and ratio of 1 (inactive firms) in tha data')
print('count: ',count_1,'rate: ', count_1/(count_1+count_0), 
      'total data :', count_1+count_0)

print('The count and ratio of 0 (active firms) in tha data')
print('count: ',count_0,'rate: ', count_0/(count_1+count_0), 
      'total data :', count_1+count_0)

# PART 1: The Statistics
# Convert the raw data as X and y
import scipy.stats as stats

# Convert data
X = data.drop(['active_inactive'],axis=1)
y = data['active_inactive']

# To create a box-plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(X)
 
# show plot
plt.show()

# Implementing Over sampling method with sampling strategy=0.5 for imbalanced data
import imblearn
from imblearn.over_sampling import RandomOverSampler

oversampling = RandomOverSampler(sampling_strategy=1)

from collections import Counter
#transform the dataset
X, Y = oversampling.fit_resample(X,y)
counter = Counter(Y)

# after implementing over sampling method we need to shuffle the data
# because the new samples are added to the last row.
X['Y']=Y
X=X.sample(frac = 1)

# We need to reset index after shuffling
y = X['Y']
X = X.drop(['Y'],axis=1)

X=X.reset_index()
X=X.drop(['index'],axis=1)

y=y.reset_index()
y=y['Y']

# The one-way ANOVA tests: The null hypothesis that two or more groups 
# have the same population mean.
fvalue, pvalue = stats.f_oneway(X['X1'], X['X2'], X['X3'], X['X4'], X['X5'])
print(fvalue, pvalue)

if pvalue<0.05:
    print(
    "p-value: {}, The null hypothesis is rejected. There is a difference" 
    "between at least two variables.".format(pvalue))
else:
    print(
    "p-value: {}, The null hypothesis is accepted. There are not any dif-"
    "ferences among the means of variables.".format(pvalue))

# // The Test of Normality //
# The null hypothesis that the input data is not from a normal distribution.
def normality(x):
    k2, pvalue = stats.normaltest(x)
    alpha = 1e-3
    
    if pvalue < alpha: # null hypothesis: x comes from a normal distribution
        return('The null hypothesis is rejected. The input data is from' 
               'a normal distribution')
    else:
        return('The null hypothesis is accepted. The input data is not from'
               'a normal distribution') 
    
print(normality(X['X1']))
print(normality(X['X2']))
print(normality(X['X3']))
print(normality(X['X4']))
print(normality(X['X5']))

import scipy.stats as stats
# // Test of Homogeneity of Variances // 
# The Levene's test is used instead of Bartlett’s test because 
# our data is not from a normal distribution.
# The null hypothesis that all input samples are from populations 
# with equal variances.
stat, p = stats.levene(X['X1'],X['X2'], X['X3'],X['X4'], X['X5'])
print(stat, p)

if pvalue<0.05:
    print(
    "p-value: {}, The null hypothesis is rejected. Not all input samples" 
    "are from populations with equal variances.".format(pvalue))
else:
    print(
    "p-value: {}, The null hypothesis is accepted. All input samples are" 
    "from populations with equal variances.".format(pvalue))

# PART 2: The Neural Net training
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

n=7775
a=0;     b=(4*n/5)-1; c=4*n/5 ; d=n-1    # Fold 1
#a=n/5;   b=n-1; c=0 ; d=n/5-1              # Fold 2
#a=2*n/5;  b=n/5-1; c=n/5 ; d=2*n/5-1         # Fold 3              
#a=3*n/5; b=2*n/5-1; c=2*n/5 ; d=3*n/5-1    # Fold 4
#a=4*n/5; b=3*n/5-1; c=3*n/5 ; d=4*n/5-1  # Fold 5

#list1 = list(range(6220,7775))
#list2 = list(range(0,4665))
#list3 = list(range(4665,6220))

#X = X.reindex(list1 + list2 + list3) 
#y = y.reindex(list1 + list2 + list3)

X_train, X_test, y_train, y_test = [], [], [], []
X_train, X_test = X.loc[a:b], X.loc[c:d]
y_train, y_test = y.loc[a:b], y.loc[c:d]

#Normalize the data using training set statistics
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train -= mean
X_test -= mean
X_train /= std
X_test /= std

count_inactive_test, count_active_test = 0, 0
for j in y_test:
    if j==1:
        count_inactive_test+=1

    if j==0:
        count_active_test+=1
        
print("inactive firms in the test:", count_inactive_test)
print("active firms in the test:", count_active_test)

# Analyze class imbalance in the targets
# 0 and 1 mean inactive, active firms respectively.
counts_1 = y_train.sum()
counts_0 = len(y_train) - counts_1


# Build a classification model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(6, activation="relu",use_bias=True),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.summary()

# define the keras model
metrics = [
    keras.metrics.Precision(name="precision"),
    keras.metrics.BinaryAccuracy(name="binary_accuracy"),
]

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss='binary_crossentropy', metrics=[metrics]
)

callbacks = [keras.callbacks.EarlyStopping(
    monitor="val_precision",
    min_delta=0,
    patience=25,
    verbose=0,
    mode="max",
    baseline=None,
    restore_best_weights=True,
)]

model.fit(
    X_train,
    y_train,
    batch_size=2,
    epochs=10000,
    verbose=1,
    callbacks=callbacks,
    validation_data=(X_test, y_test)
)

# Generate generalization metrics
scores_test = model.evaluate(X_test, y_test, verbose=0)
scores_train = model.evaluate(X_train, y_train, verbose=0)
print(f'Score for fold: {model.metrics_names[0]} of {scores_test[0]};%')
print(f'Score for fold: {model.metrics_names[0]} of {scores_train[0]};%')

# Fold 1
weights = []
for layer in model.layers:
    weights.append(layer.get_weights())
    
weights

import seaborn as sns

y_pred=model.predict(X_test)

classes_x=[]
for y in y_pred:
    if y>=0.5:
        classes_x.append(1)
    else:
        classes_x.append(0)

sns.distplot(classes_x)
print(model.evaluate(X_test, y_test))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, classes_x)

# Fold 2
weights = []
for layer in model.layers:
    weights.append(layer.get_weights())
    
weights

y_pred=model.predict(X_test)

classes_x=[]
for y in y_pred:
    if y>=0.5:
        classes_x.append(1)
    else:
        classes_x.append(0)

sns.distplot(classes_x)
print(model.evaluate(X_test, y_test))

confusion_matrix(y_test, classes_x)

# Fold 3
weights = []
for layer in model.layers:
    weights.append(layer.get_weights())
    
weights

y_pred=model.predict(X_test)

classes_x=[]
for y in y_pred:
    if y>=0.5:
        classes_x.append(1)
    else:
        classes_x.append(0)

sns.distplot(classes_x)
print(model.evaluate(X_test, y_test))

confusion_matrix(y_test, classes_x)

# Fold 4
weights = []
for layer in model.layers:
    weights.append(layer.get_weights())
    
weights

y_pred=model.predict(X_test)

classes_x=[]
for y in y_pred:
    if y>=0.5:
        classes_x.append(1)
    else:
        classes_x.append(0)

sns.distplot(classes_x)
print(model.evaluate(X_test, y_test))

confusion_matrix(y_test, classes_x)

# Fold 5
weights = []
for layer in model.layers:
    weights.append(layer.get_weights())
    
weights

y_pred=model.predict(X_test)

classes_x=[]
for y in y_pred:
    if y>=0.5:
        classes_x.append(1)
    else:
        classes_x.append(0)

sns.distplot(classes_x)
print(model.evaluate(X_test, y_test))

confusion_matrix(y_test, classes_x)