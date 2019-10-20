import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# To decide display window width on console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 30)


def read_data(folder_path):
    data = pd.read_csv(folder_path, sep=',')
    return data


# Read the loan prediction dataset
train_data = read_data("data/train.csv")
test_data = read_data("data/test.csv")
# print("Some sample of training set:\n{}".format(train_data.sample(n=10)))
# print("Some sample of test set:\n{}".format(test_data.sample(n=10)))

# Let us visualize the distribution of all important features
plt.figure(1, figsize=(5, 5))
plt.tight_layout()
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
plt.show(block=False)

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

plt.show(block=False)

# Clean the dataset
train_data['Dependents'].replace('3+', 3, inplace=True)
test_data['Dependents'].replace('3+', 3, inplace=True)
enc = LabelEncoder()
y_train = enc.fit_transform(train_data['Loan_Status'])
X_train = train_data.iloc[:, :-1]
X_test = test_data

# print(X_train.isnull().sum())

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

# Remove Loan_ID
X_train = X_train.drop('Loan_ID', axis=1)
X_test = X_test.drop('Loan_ID', axis=1)

# Normalize all numerical features using MinMax Scalar
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
scalar = MinMaxScaler()
X_train[numerical_columns] = scalar.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scalar.fit_transform(X_test[numerical_columns])

# One hot encode all categorical feature columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

X_train_enc = pd.concat([X_train, pd.get_dummies(X_train[categorical_columns])], axis=1)
X_train_enc = X_train_enc.drop(categorical_columns, axis=1)

X_test_enc = pd.concat([X_test, pd.get_dummies(X_test[categorical_columns])], axis=1)
X_test_enc = X_test_enc.drop(categorical_columns, axis=1)

# Split the training set into train and cross validation set
trainX, cvX, trainY, cvY = train_test_split(X_train_enc, y_train, test_size=.1,
                                            stratify=y_train, random_state=42)

# Implement Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(trainX, trainY)

print("Logistic Regression Accuracy Score: %s" % log_reg.score(cvX, cvY))

# With Logistic Regression 10-Fold Cross validarion
i = 1
kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X_train_enc, y_train):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X_train_enc.loc[train_index], X_train_enc.loc[test_index]
    ytr, yvl = y_train[train_index], y_train[test_index]

    model = LogisticRegression(random_state=0)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
    if i > 10:
        prediction = model.predict(X_test_enc)
        print('\n\nPrediction on Test Set:\n{}'.format(enc.inverse_transform(prediction)))