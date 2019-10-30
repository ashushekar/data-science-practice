import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

warnings.filterwarnings('ignore')

# To decide display window width on console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 30)


red_wine_data = pd.read_csv("data/winequality-red.csv", sep=';')
# print(red_wine_data.describe())

features = red_wine_data.iloc[:, 1:len(red_wine_data.columns) -1]
labels = red_wine_data.iloc[:, -1]
scalar = MinMaxScaler()

column_names = [columns.replace(" ", "-") for columns in features.columns]
features = pd.DataFrame(scalar.fit_transform(features), columns=column_names)

# print(features.sample(n=10))

print("----------Correlation Matrix----------")
features['labels'] = labels
correlation_matrix = features.corr()
print(correlation_matrix)
top_features = correlation_matrix.index

# Generate a mask for the upper triangle
mask = np.zeros_like(correlation_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(features[top_features].corr(), mask=mask, cmap=cmap, vmax=1.0, vmin=0.0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show(block=True)

# features = features.drop('labels', axis=1, inplace=True)

encoder = OneHotEncoder()
labels = features['labels'].values
labels_enc = encoder.fit_transform(labels.reshape(-1, 1))
print(labels_enc)
features = features.drop('labels', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(features, labels_enc, test_size=0.1)

classifier = SVC(kernel='linear')
rfe = RFE(estimator=classifier, n_features_to_select=1)
rfe.fit(X_train, y_train)
print(rfe.ranking_)
