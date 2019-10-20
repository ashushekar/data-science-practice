import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# read the data and set the column names
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_table('iris.data', sep=',', header=None)
data.columns = column_names

print(data.sample(n=5))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['class'],
                                                    stratify=data['class'], random_state=42)

# Now let us select important feature using SelectPercentile which is a
# univariate-anova-selection approach

select = SelectPercentile(percentile=75)
select.fit(X_train, y_train)

X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
print("Shape of features before feature selection: {}".format(X_train.shape))
print("Shape of features after feature selection: {}".format(X_train_selected.shape))

# Label encode the labels
enc = LabelEncoder()
y_train_enc = enc.fit_transform(y_train)
y_test_enc = enc.fit_transform(y_test)

# Now apply Logistic regression on transformed set
log_reg = LogisticRegression()

log_reg.fit(X_train_selected, y_train_enc)
prediction = log_reg.predict(X_test_selected)
pred_labels = enc.inverse_transform(prediction)

print("Classification Report for logistic Regression:\n")
print(classification_report(y_test, pred_labels, labels=list(set(y_train))))
labels = list(set(y_train))
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix(y_test, pred_labels, labels=labels))
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=True)


# Apply SVM
svm_classifier = SVC(kernel='linear', C=0.3)
svm_classifier.fit(X_train_selected, y_train_enc)
prediction = svm_classifier.predict(X_test_selected)
pred_labels = enc.inverse_transform(prediction)

print("\nClassification Report for SVM Classifier:\n")
print(classification_report(y_test, pred_labels, labels=list(set(y_train))))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix(y_test, pred_labels, labels=labels))
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=True)


