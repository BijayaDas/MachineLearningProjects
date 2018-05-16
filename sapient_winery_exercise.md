

```python
#This is a standard classification problem.
#out of lot of classifiers I have used Logistic Regression, Decision Tree and experimented with MLP Neural Network classifier.

#Assumptions: Red and White are replaced in the dataset preprocessing as 1 and 2 respectively.
#Acuracy Scores : ~52% using Logistic Regression, ~60% using Decision Tree and <50% using Neural Network.(Which needs improvement by fine tuning the model)

#NOTE: These models can be improved and fine tuned given time and expertise.
```


```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
```


```python
data = pd.read_csv('wine_dataset.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
```

    (6497, 13)
    ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'style']



```python
mapping = {'red': 1, 'white': 2}
data = data.replace({'style': mapping})
```


```python
X = data.drop('quality', axis=1)
y = data.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train)
```

          fixed_acidity  volatile_acidity  citric_acid  residual_sugar  chlorides  \
    3731            7.1             0.220         0.32           16.90      0.056   
    1651            6.2             0.160         0.33            1.10      0.057   
    5060            6.7             0.240         0.30            3.85      0.042   
    1303            8.0             0.280         0.44            1.80      0.081   
    474            10.5             0.280         0.51            1.70      0.080   
    4304            6.7             0.260         0.39            6.40      0.171   
    2731            6.8             0.390         0.31           14.35      0.043   
    1855            6.3             0.350         0.30            5.70      0.035   
    2752            5.3             0.395         0.07            1.30      0.035   
    5137            6.5             0.280         0.25            4.80      0.029   
    567             8.7             0.700         0.24            2.50      0.226   
    1949            6.3             0.120         0.36            2.10      0.044   
    2267            6.1             0.140         0.25            1.30      0.047   
    4053            7.1             0.290         0.30           16.00      0.036   
    2089            7.0             0.280         0.36            1.00      0.035   
    5776            6.4             0.250         0.33            1.70      0.037   
    6320            6.1             0.320         0.33           10.70      0.036   
    3900            6.6             0.220         0.28           12.05      0.058   
    1979            7.3             0.200         0.44            1.40      0.045   
    4205            7.2             0.200         0.36            2.50      0.028   
    666             8.3             0.490         0.36            1.80      0.222   
    3655            6.8             0.210         0.27           18.15      0.042   
    5656            6.6             0.170         0.36            1.90      0.036   
    3943            7.0             0.230         0.26            7.20      0.041   
    2126            6.1             0.280         0.22            1.80      0.034   
    6256            6.0             0.290         0.41           10.80      0.048   
    4663            7.4             0.210         0.80           12.30      0.038   
    5064            6.8             0.190         0.34            1.90      0.040   
    5329            6.2             0.220         0.20           20.80      0.035   
    2527            6.5             0.250         0.35           12.00      0.055   
    ...             ...               ...          ...             ...        ...   
    99              8.1             0.545         0.18            1.90      0.080   
    2496            7.0             0.290         0.26            1.60      0.044   
    1871            5.8             0.250         0.26           13.10      0.051   
    2046            6.5             0.410         0.24           14.00      0.048   
    4851            7.1             0.260         0.37            5.50      0.025   
    5072            6.5             0.300         0.27            4.00      0.038   
    2163            6.8             0.510         0.30            4.20      0.066   
    6036            6.5             0.290         0.30            9.15      0.051   
    6216            5.2             0.500         0.18            2.00      0.036   
    2893            6.9             0.410         0.22            4.20      0.031   
    537             8.1             0.825         0.24            2.10      0.084   
    1701            6.0             0.210         0.24           12.10      0.050   
    2897            7.3             0.340         0.39            5.20      0.040   
    2222            7.2             0.240         0.34            1.10      0.045   
    2135            7.9             0.345         0.51           15.30      0.047   
    2599            8.0             0.190         0.36            1.80      0.050   
    705             8.4             1.035         0.15            6.00      0.073   
    6458            6.0             0.430         0.34            7.60      0.045   
    3468            8.2             0.180         0.28            8.50      0.035   
    5924            6.4             0.240         0.26            8.20      0.054   
    5874            5.7             0.220         0.20           16.00      0.044   
    4373            7.0             0.360         0.32           10.05      0.045   
    1033            7.5             0.570         0.08            2.60      0.089   
    5827            6.2             0.290         0.23           12.40      0.048   
    4859            7.4             0.190         0.31           14.50      0.045   
    4931            6.5             0.220         0.28            3.70      0.059   
    3264            6.5             0.130         0.37            1.00      0.036   
    1653            6.8             0.200         0.59            0.90      0.147   
    2607            6.6             0.220         0.37            1.20      0.059   
    2732            8.7             0.220         0.42            2.30      0.053   
    
          free_sulfur_dioxide  total_sulfur_dioxide  density    pH  sulphates  \
    3731                 49.0                 158.0  0.99980  3.37       0.38   
    1651                 21.0                  82.0  0.99100  3.32       0.46   
    5060                105.0                 179.0  0.99189  3.04       0.59   
    1303                 28.0                  68.0  0.99501  3.36       0.66   
    474                  10.0                  24.0  0.99820  3.20       0.89   
    4304                 64.0                 200.0  0.99562  3.19       0.38   
    2731                 28.0                 162.0  0.99880  3.17       0.54   
    1855                  8.0                  97.0  0.99270  3.27       0.41   
    2752                 26.0                 102.0  0.99200  3.50       0.35   
    5137                 54.0                 128.0  0.99074  3.17       0.44   
    567                   5.0                  15.0  0.99910  3.32       0.60   
    1949                 47.0                 146.0  0.99140  3.27       0.74   
    2267                 37.0                 173.0  0.99250  3.35       0.46   
    4053                 58.0                 201.0  0.99954  3.30       0.67   
    2089                  8.0                  70.0  0.98990  3.09       0.46   
    5776                 35.0                 113.0  0.99164  3.23       0.66   
    6320                 27.0                  98.0  0.99521  3.34       0.52   
    3900                 25.0                 125.0  0.99856  3.45       0.45   
    1979                 21.0                  98.0  0.99240  3.15       0.46   
    4205                 22.0                 157.0  0.99380  3.48       0.49   
    666                   6.0                  16.0  0.99800  3.18       0.60   
    3655                 41.0                 146.0  1.00010  3.30       0.36   
    5656                 38.0                 110.0  0.99056  3.05       0.54   
    3943                 21.0                  90.0  0.99509  3.22       0.55   
    2126                 32.0                 116.0  0.98980  3.36       0.44   
    6256                 55.0                 149.0  0.99370  3.09       0.59   
    4663                 77.0                 183.0  0.99778  2.95       0.48   
    5064                 41.0                 108.0  0.99000  3.25       0.45   
    5329                 58.0                 184.0  1.00022  3.11       0.53   
    2527                 47.0                 179.0  0.99800  3.58       0.47   
    ...                   ...                   ...      ...   ...        ...   
    99                   13.0                  35.0  0.99720  3.30       0.59   
    2496                 12.0                  87.0  0.99230  3.08       0.46   
    1871                 44.0                 148.0  0.99720  3.29       0.38   
    2046                 24.0                 113.0  0.99820  3.44       0.53   
    4851                 31.0                 105.0  0.99082  3.06       0.33   
    5072                 37.0                  97.0  0.99026  3.20       0.60   
    2163                 38.0                 165.0  0.99450  3.20       0.42   
    6036                 25.0                 166.0  0.99339  3.24       0.56   
    6216                 23.0                 129.0  0.98949  3.36       0.77   
    2893                 10.0                 102.0  0.99300  3.00       0.86   
    537                   5.0                  13.0  0.99720  3.37       0.77   
    1701                 55.0                 164.0  0.99700  3.34       0.39   
    2897                 45.0                 163.0  0.99250  3.30       0.47   
    2222                  3.0                  64.0  0.99130  3.23       0.51   
    2135                 54.0                 171.0  0.99870  3.09       0.51   
    2599                 16.0                  84.0  0.99360  3.15       0.45   
    705                  11.0                  54.0  0.99900  3.37       0.49   
    6458                 25.0                 118.0  0.99222  3.03       0.37   
    3468                 41.0                 140.0  0.99520  3.04       0.37   
    5924                 47.0                 182.0  0.99538  3.12       0.50   
    5874                 41.0                 113.0  0.99862  3.22       0.46   
    4373                 37.0                 131.0  0.99352  3.09       0.33   
    1033                 14.0                  27.0  0.99592  3.30       0.59   
    5827                 33.0                 201.0  0.99612  3.11       0.56   
    4859                 39.0                 193.0  0.99860  3.10       0.50   
    4931                 29.0                 151.0  0.99177  3.23       0.41   
    3264                 48.0                 114.0  0.99110  3.41       0.51   
    1653                 38.0                 132.0  0.99300  3.05       0.38   
    2607                 45.0                 199.0  0.99300  3.37       0.55   
    2732                 27.0                 114.0  0.99400  2.99       0.43   
    
          alcohol  style  
    3731     9.60      2  
    1651    10.90      2  
    5060    11.30      2  
    1303    11.20      1  
    474      9.40      1  
    4304     9.40      2  
    2731     9.10      2  
    1855    11.00      2  
    2752    10.60      2  
    5137    12.20      2  
    567      9.00      1  
    1949    11.40      2  
    2267    10.00      2  
    4053     9.00      2  
    2089    12.10      2  
    5776    10.60      2  
    6320    10.20      2  
    3900     9.40      2  
    1979    10.00      2  
    4205    10.60      2  
    666      9.50      1  
    3655     8.70      2  
    5656    11.40      2  
    3943     9.50      2  
    2126    12.60      2  
    6256    11.00      2  
    4663     9.00      2  
    5064    12.90      2  
    5329     9.00      2  
    2527    10.00      2  
    ...       ...    ...  
    99       9.00      1  
    2496    10.50      2  
    1871     9.30      2  
    2046     9.80      2  
    4851    12.60      2  
    5072    12.60      2  
    2163     9.10      2  
    6036    11.35      2  
    6216    13.40      2  
    2893    11.60      2  
    537     10.70      1  
    1701     9.40      2  
    2897    12.40      2  
    2222    11.40      2  
    2135     9.10      2  
    2599     9.80      2  
    705      9.90      1  
    6458    11.00      2  
    3468    10.10      2  
    5924     9.50      2  
    5874     8.90      2  
    4373    11.70      2  
    1033    10.40      1  
    5827     9.90      2  
    4859     9.20      2  
    4931    12.10      2  
    3264    11.50      2  
    1653     9.10      2  
    2607    10.30      2  
    2732    10.00      2  
    
    [4872 rows x 12 columns]



```python
print(y_train)
```

    3731    6
    1651    7
    5060    8
    1303    5
    474     6
    4304    6
    2731    5
    1855    7
    2752    6
    5137    7
    567     6
    1949    7
    2267    6
    4053    5
    2089    6
    5776    6
    6320    6
    3900    5
    1979    7
    4205    6
    666     6
    3655    5
    5656    6
    3943    6
    2126    6
    6256    7
    4663    5
    5064    6
    5329    6
    2527    5
           ..
    99      6
    2496    6
    1871    5
    2046    6
    4851    8
    5072    8
    2163    5
    6036    6
    6216    7
    2893    4
    537     6
    1701    5
    2897    6
    2222    5
    2135    5
    2599    7
    705     5
    6458    6
    3468    7
    5924    5
    5874    6
    4373    8
    1033    6
    5827    6
    4859    6
    4931    7
    3264    8
    1653    6
    2607    7
    2732    5
    Name: quality, Length: 4872, dtype: int64



```python
X_train.shape
```




    (4872, 12)




```python
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
```

    [[  0   0   6   4   0   0   0]
     [  0   0  39  16   0   0   0]
     [  0   0 296 235   1   0   0]
     [  0   0 175 521   8   0   0]
     [  0   0  20 242  12   0   0]
     [  0   0   2  44   1   0   0]
     [  0   0   0   3   0   0   0]]



```python
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
```

    Accuracy of logistic regression classifier on test set: 0.51



```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

                 precision    recall  f1-score   support
    
              3       0.00      0.00      0.00        10
              4       0.00      0.00      0.00        55
              5       0.55      0.56      0.55       532
              6       0.49      0.74      0.59       704
              7       0.55      0.04      0.08       274
              8       0.00      0.00      0.00        47
              9       0.00      0.00      0.00         3
    
    avg / total       0.48      0.51      0.45      1625
    


    /Users/bkdas/anaconda/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
prediction = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(prediction)
```

    
    The confidence score:
    
    0.5963076923076923



```python
y_pred = clf.predict(X_test)
```


```python
#converting the numpy array to list
j=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print(j[i])
    
#printing first five expectations
print("\nThe expectation:\n")
print(y_test.head())
```

    
    The prediction:
    
    6
    6
    6
    4
    8
    
    The expectation:
    
    5316    6
    5210    6
    3518    6
    1622    5
    2443    8
    Name: quality, dtype: int64



```python
clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
```


```python
clf_mlp.fit(X_train, y_train)
```




    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)




```python
nn_predictions = clf_mlp.predict(X_test)
```


```python
print(nn_predictions)
```

    [6 6 5 ... 6 6 6]



```python
print(classification_report(y_test,nn_predictions))
```

                 precision    recall  f1-score   support
    
              3       0.00      0.00      0.00        10
              4       0.00      0.00      0.00        55
              5       0.48      0.32      0.38       532
              6       0.45      0.72      0.55       704
              7       0.40      0.20      0.27       274
              8       0.00      0.00      0.00        47
              9       0.00      0.00      0.00         3
    
    avg / total       0.42      0.45      0.41      1625
    


    /Users/bkdas/anaconda/lib/python3.6/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python

```
