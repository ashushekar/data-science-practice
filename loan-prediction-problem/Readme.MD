# Loan Prediction Analysis
We will do some preprocessing on Loan Prediction dataset before proceeding with any kind of training on the data. 

## Dataset
Loan Prediction dataset contains complete loan data for all loans issued through the 2007-2015, including the current 
loan status (Current, Late, Fully Paid, etc.) and latest payment information. The file containing loan data through the 
"present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional 
features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among 
others. The file is a matrix of about 890 thousand observations and 75 variables. A data dictionary is provided in a 
separate file.

#### Import all Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# To decide display window width on console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 30)
```
#### Import the dataset

```python
def read_data(folder_path):
    data = pd.read_csv(folder_path, sep=',')
    return data


# Read the loan prediction dataset
train_data = read_data("data/train.csv")
test_data = read_data("data/test.csv")
print("Some sample of training set:\n{}".format(train_data.sample(n=10)))
# print("Some sample of test set:\n{}".format(test_data.sample(n=10)))
```

```sh 
Some sample of training set:
      Loan_ID  Gender Married Dependents     Education Self_Employed  ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History Property_Area Loan_Status
133  LP001482    Male     Yes          0      Graduate           Yes             3459                0.0        25.0             120.0             1.0     Semiurban           Y
506  LP002624    Male     Yes          0      Graduate            No            20833             6667.0       480.0             360.0             NaN         Urban           Y
354  LP002143  Female     Yes          0      Graduate            No             2423              505.0       130.0             360.0             1.0     Semiurban           Y
115  LP001401    Male     Yes          1      Graduate            No            14583                0.0       185.0             180.0             1.0         Rural           Y
500  LP002603  Female      No          0      Graduate            No              645             3683.0       113.0             480.0             1.0         Rural           Y
271  LP001891    Male     Yes          0      Graduate            No            11146                0.0       136.0             360.0             1.0         Urban           Y
305  LP001990    Male      No          0  Not Graduate            No             2000                0.0         NaN             360.0             1.0         Urban           N
501  LP002606  Female      No          0      Graduate            No             3159                0.0       100.0             360.0             1.0     Semiurban           Y
486  LP002545    Male      No          2      Graduate            No             3547                0.0        80.0             360.0             0.0         Rural           N
196  LP001666    Male      No          0      Graduate            No             8333             3750.0       187.0             360.0             1.0         Rural           Y
```

#### Visualize the Dataset

Let us understand through visualization the important features present in the dataset and correlate between features.
```python
plt.figure(1)
plt.subplot(321)
train_data['Loan_Status'].value_counts(normalize=True).plot.bar(title='Loan Status')
plt.subplot(322)
train_data['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
plt.subplot(323)
train_data['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(324)
train_data['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.subplot(325)
train_data['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(326)
train_data['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show(block=True)
``` 

From the plots it is clear that number of loans given were successful. Also the number of graduates is more than number of
non-graduates. The dataset has number of males more than number of females.

Let us plot few more comparison graphs in which we can check the loan status significantly changes because of which feature.
```python
genderLoanStatus = pd.crosstab(train_data['Gender'], train_data['Loan_Status'])
genderLoanStatus.div(genderLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Gender vs Loan Status',
                                                                             stacked=True)

marriedLoanStatus = pd.crosstab(train_data['Married'], train_data['Loan_Status'])
marriedLoanStatus.div(marriedLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Married vs Loan Status',
                                                                               stacked=True)

dependentsLoanStatus = pd.crosstab(train_data['Dependents'], train_data['Loan_Status'])
dependentsLoanStatus.div(dependentsLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Dependents vs Loan Status',
                                                                                     stacked=True)

educationLoanStatus = pd.crosstab(train_data['Education'], train_data['Loan_Status'])
educationLoanStatus.div(educationLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Education vs Loan Status',
                                                                                   stacked=True)

selfemployedLoanStatus = pd.crosstab(train_data['Self_Employed'], train_data['Loan_Status'])
selfemployedLoanStatus.div(selfemployedLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Self-Employed vs Loan Status',
                                                                                         stacked=True)

creditLoanStatus = pd.crosstab(train_data['Credit_History'], train_data['Loan_Status'])
creditLoanStatus.div(creditLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Credit-loan vs Loan Status',
                                                                             stacked=True)

propertyLoanStatus = pd.crosstab(train_data['Property_Area'], train_data['Loan_Status'])
propertyLoanStatus.div(propertyLoanStatus.sum(1).astype(float), axis=0).plot.bar(title='Property-Area vs Loan Status',
                                                                                 stacked=True)

plt.show(block=True)
```
![educationvsloan](https://user-images.githubusercontent.com/35737777/67167584-b1308580-f393-11e9-9167-49a5d0163c08.png)

![gendervsloan](https://user-images.githubusercontent.com/35737777/67167585-b1308580-f393-11e9-90f3-68ad04a1f043.png)

![marriedvsloan](https://user-images.githubusercontent.com/35737777/67167586-b1308580-f393-11e9-9235-e627b4306b24.png)

![propertyvsloan](https://user-images.githubusercontent.com/35737777/67167587-b1308580-f393-11e9-9996-28b2be29eafc.png)

![selfemployedvsloan](https://user-images.githubusercontent.com/35737777/67167588-b1c91c00-f393-11e9-96e9-ba1b5e633088.png)

![all_counts_plot](https://user-images.githubusercontent.com/35737777/67167589-b1c91c00-f393-11e9-98d6-106b872fa844.png)

![creditvsloan](https://user-images.githubusercontent.com/35737777/67167590-b1c91c00-f393-11e9-8717-003a43c16ce3.png)

![dependentsvsloan](https://user-images.githubusercontent.com/35737777/67167591-b1c91c00-f393-11e9-8a2f-b1caad3e2a83.png)

From the graph, the plot of credit-history vs loan-status shows more detailed information. It indicates that possibility
of loan approval as Successful depends on credit-history. If credit-history is None then loan status is No.

### Clean the dataset and preprocess

The Dependents column has 3+ value in it, indicating Number of dependents has more than 3. We will approximate it to 3.
Label Encode the Loan Status column.

```python
enc = LabelEncoder()
y_train = enc.fit_transform(train_data['Loan_Status'])
X_train = train_data.iloc[:, :-1]
X_test = test_data
```

Check for Null values, and if present replace it with _mode_ values 
```python
print(X_train.isnull().sum())
```
```sh 
Loan_ID               0
Gender               13
Married               3
Dependents           15
Education             0
Self_Employed        32
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount           22
Loan_Amount_Term     14
Credit_History       50
Property_Area         0
dtype: int64
```

```python
# Replace Nan values with MODE values for both train/test set
X_train['Gender'].fillna(X_train['Gender'].mode()[0], inplace=True)
X_train['Married'].fillna(X_train['Married'].mode()[0], inplace=True)
X_train['Dependents'].fillna(X_train['Dependents'].mode()[0], inplace=True)
X_train['Self_Employed'].fillna(X_train['Self_Employed'].mode()[0], inplace=True)
X_train['LoanAmount'].fillna(X_train['LoanAmount'].mode()[0], inplace=True)
X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mode()[0], inplace=True)
X_train['Credit_History'].fillna(X_train['Credit_History'].mode()[0], inplace=True)
# print(X_train.isnull().sum())

# print(X_test.isnull().sum())

X_test['Gender'].fillna(X_test['Gender'].mode()[0], inplace=True)
X_test['Married'].fillna(X_test['Married'].mode()[0], inplace=True)
X_test['Dependents'].fillna(X_test['Dependents'].mode()[0], inplace=True)
X_test['Self_Employed'].fillna(X_test['Self_Employed'].mode()[0], inplace=True)
X_test['LoanAmount'].fillna(X_test['LoanAmount'].mode()[0], inplace=True)
X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mode()[0], inplace=True)
X_test['Credit_History'].fillna(X_test['Credit_History'].mode()[0], inplace=True)

# print(X_test.isnull().sum())
```

We do not Loan_ID column for feature training. We will remove it
```python
# Remove Loan_ID
X_train = X_train.drop('Loan_ID', axis=1)
X_test = X_test.drop('Loan_ID', axis=1)
```

Some attributes seems to be categorical columns. It is better to one hot encode all those categorical features.
```python
# One hot encode all categorical feature columns
categorical_columns = X_train.select_dtypes(include=['object']).columns
X_train_enc = pd.concat([X_train, pd.get_dummies(X_train[categorical_columns])], axis=1)
X_train_enc = X_train_enc.drop(categorical_columns, axis=1)

X_test_enc = pd.concat([X_test, pd.get_dummies(X_test[categorical_columns])], axis=1)
X_test_enc = X_test_enc.drop(categorical_columns, axis=1)
```

#### Train-Test Split

Split the training set into X and Y set.
```python
trainX, cvX, trainY, cvY = train_test_split(X_train_enc, y_train, test_size=.1,
                                            stratify=y_train, random_state=42)
```

## Train the model
### Logistic Regression
#### Without K-Fold Cross Validation
```python
log_reg = LogisticRegression()
log_reg.fit(trainX, trainY)

print("Logistic Regression Accuracy Score: %s" % log_reg.score(cvX, cvY))
``` 

```sh 
Logistic Regression Accuracy Score: 0.8225806451612904
```
#### With 10-fold Cross validation
```python
i = 1
kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X_train_enc, y_train):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X_train_enc.loc[train_index], X_train_enc.loc[test_index]
    ytr, yvl = y_train[train_index], y_train[test_index]

    model = LogisticRegression()
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
    if i > 10:
        prediction = model.predict(X_test_enc)
        print('Prediction on Test Set:\n{}'.format(enc.inverse_transform(prediction)))
```

```sh 
1 of kfold 10
accuracy_score 0.7777777777777778

2 of kfold 10
accuracy_score 0.8095238095238095

3 of kfold 10
accuracy_score 0.7868852459016393

4 of kfold 10
accuracy_score 0.8360655737704918

5 of kfold 10
accuracy_score 0.8032786885245902

6 of kfold 10
accuracy_score 0.819672131147541

7 of kfold 10
accuracy_score 0.8032786885245902

8 of kfold 10
accuracy_score 0.7868852459016393

9 of kfold 10
accuracy_score 0.8360655737704918

10 of kfold 10
accuracy_score 0.8360655737704918


Prediction on Test Set:
['Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y'
 'Y' 'N' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'N' 'N' 'Y' 'Y' 'Y' 'N' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y'
 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'N' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'N' 'Y' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'N'
 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'N' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'N' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y'
 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y']
```
