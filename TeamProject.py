import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, roc_curve, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier


warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", 25, "display.max_columns", 24)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

dataset = pd.read_csv('C:\weatherAUS.csv')

# statistical summary of dataset
print("# dataset statistical")
print(dataset.describe(), "\n")
print("# dataset head")
print(dataset.head(5), "\n")
print("# dataset shape")
print(dataset.shape, "\n")
print("# dataset features")
print(dataset.columns, "\n")
print("# dataset informations")
print(dataset.info(), "\n")

# plot a line chart of Min/Max temp in the latest years
dataset['Month'] = dataset['Date'].str.slice(start=5, stop=7)
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%Y/%m/%d', errors='ignore')
dataset_dateplot = dataset.iloc[-900:, :]
plt.figure(figsize=[20, 3])
plt.plot(dataset_dateplot['Date'], dataset_dateplot['MinTemp'], color='blue')
plt.plot(dataset_dateplot['Date'], dataset_dateplot['MaxTemp'], color='red')
plt.fill_between(dataset_dateplot['Date'], dataset_dateplot['MinTemp'], dataset_dateplot['MaxTemp'],
                 facecolor='#EBF78F')
#plt.legend()
plt.show()

# convert object(Yes/No) to binary
dataset['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
dataset['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# drop RISK_MM as said in the dataset description
dataset = dataset.drop('RISK_MM', axis=1)
# drop Date because it is useless
dataset = dataset.drop('Date', axis=1)

print("# find which column have object variables")
print(dataset.select_dtypes(include=['object']).columns)

dataset['Location'] = dataset['Location'].fillna(dataset['Location'].mode()[0])
dataset['WindGustDir'] = dataset['WindGustDir'].fillna(dataset['WindGustDir'].mode()[0])
dataset['WindDir9am'] = dataset['WindDir9am'].fillna(dataset['WindDir9am'].mode()[0])
dataset['WindDir3pm'] = dataset['WindDir3pm'].fillna(dataset['WindDir3pm'].mode()[0])

le = LabelEncoder()
for col in dataset.select_dtypes(include=['object']).columns:
    dataset[col] = le.fit_transform(dataset[col])

# correlation
correlation = dataset.corr()
# tick labels
matrix_cols = correlation.columns.tolist()
# convert to array
corr_array = np.array(correlation)

sns.clustermap(correlation,
               annot = True,
               cmap = 'RdYlBu_r',
               vmin = -1, vmax = 1,
               )
plt.show()

# drop the columns which correlation is -0.1 ~ 0.1
index = []
for i in range(20):
    if abs(corr_array[19][i]) < 0.1:
        index.append(matrix_cols[i])

print("")
print("# columns which have less correlation(threshold = -0.1 ~ +0.1)")
print(index)
print("--> we drop it")

for i in range(len(index)):
    dataset = dataset.drop(columns=index[i])

print("")
print("# find how many null values are in our dataset")
print(dataset.isnull().sum())
dataset.fillna(dataset.mean(), inplace=True)
print("--> fill with mean value")
print(dataset.isnull().sum())

#use minmaxScaler in order to avoid negative values
minmax = MinMaxScaler()
scaled = minmax.fit_transform(dataset)
dataset = pd.DataFrame(scaled, columns=dataset.columns)

print("")
print("--> after preprocessing ")
print(dataset.head(5))

#Feature importance using Filter Method
from sklearn.feature_selection import SelectKBest, chi2
X = dataset.loc[:,dataset.columns != 'RainTomorrow']
y = dataset[['RainTomorrow']]
selector = SelectKBest(chi2, k=10)
selector.fit(X,y)
X_new = selector.transform(X)
print("")
print("# Feature Importance using Chi-Square")
print(X.columns[selector.get_support(indices=True)])

features = dataset.drop(columns='RainTomorrow')
target = dataset['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=34)

def plot_roc_cur(fper, tper):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    if verbose == False:
        model.fit(X_train, y_train, verbose = 0)
    else :
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print(classification_report(y_test, y_pred, digits=5))

    probs = model.predict_proba(X_test)
    probs = probs[:,1]
    fper, tper, thresholds = roc_curve(y_test, probs)
    plot_roc_cur(fper, tper)

    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='all')

    return  model, accuracy, roc_auc

print("")
print("1. logistic regression by Lasso")

params_lr = {'penalty': 'l1', 'solver': 'liblinear'}
model_lr = LogisticRegression(**params_lr)
model_lr, accuracy_lr, roc_auc_lr = run_model(model_lr, X_train, y_train, X_test, y_test)

print("")
print("2. Decision Tree")

params_dt = {'max_depth' : 16, 'max_features':"sqrt"}
model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt = run_model(model_dt, X_train, y_train, X_test, y_test)

print("")
print("3. Random Forest")

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 34}

model_rf = RandomForestClassifier(**params_rf)
model_rf, accuracy_rf, roc_auc_rf = run_model(model_rf, X_train, y_train, X_test, y_test)

print("")
print("4. SVM")

params_svm = {'C': 1.0,
              'kernel':'linear',
              'max_iter':-1,
              'probability':True,
              'random_state':None}
model_svc = SVC(**params_svm)
model_svc, accuracy_svc, roc_auc_svc = run_model(model_svc, X_train, y_train, X_test, y_test)

print("")
print("5. KNN")

#find k using cross validation
k_scores = []

for i in range(1,10):
    knn = KNeighborsClassifier(i)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

k = k_scores.index(max(k_scores))
print("--> find k with 10-cross-validation = " + str(k))

params_knn = {'n_neighbors':k,
              'weights':'distance'}
model_knn = KNeighborsClassifier(**params_knn)
model_knn, accuracy_knn, roc_auc_knn = run_model(model_knn, X_train, y_train, X_test, y_test)

print("")
print("6. xgboost")

params_xgb = {'n_estimators' : 500,
              'max_depth': 16}
model_xgb = xgb.XGBClassifier(**params_xgb)
model_xgb, accuracy_xgb, roc_auc_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)

print('')
print('7. sklearn GB')
params_gb = {'n_estimators' : 300,
             'max_depth' : 8}

model_gb = GradientBoostingClassifier(**params_gb)
model_gb, accuracy_gb, roc_auc_gb = run_model(model_gb, X_train, y_train, X_test, y_test)

print('')
print('8. GaussianNB')
params_nb = {}

model_nb = GradientBoostingClassifier(**params_gb)
model_nb, accuracy_nb, roc_auc_nb = run_model(model_nb, X_train, y_train, X_test, y_test)

models_list=[
    ('logistic', model_lr),
    ('dt', model_dt),
    ('rf', model_rf ),
    ('svm', model_svc),
    ('knn', model_knn),
    ('xgb', model_xgb),
    ('gb', model_gb),
    ('nb', model_nb)
]

model_vote = VotingClassifier(models_list, voting='soft')

model_vote, accuracy_vote, roc_auc_vote = run_model(model_vote, X_train, y_train, X_test, y_test)

accuracy_list=[accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svc, accuracy_knn, accuracy_xgb, accuracy_gb, accuracy_nb, accuracy_vote]
auc_list=[roc_auc_lr, roc_auc_dt, roc_auc_rf, roc_auc_svc, roc_auc_knn, roc_auc_xgb, roc_auc_gb, roc_auc_nb, roc_auc_vote]

scores = pd.DataFrame(data=None, index=['logistic', 'dt', 'rf', 'svd', 'knn', 'xgb', 'gb', 'nb', 'vote'], columns=['accuracy', 'auc'])
scores['accuracy'] = accuracy_list
scores['auc'] = auc_list

print(f"Best Accuracy : {dict(scores.sort_values(by='accuracy', ascending=False).head(1)['accuracy'])}")
print(f"Best AUC : {dict(scores.sort_values(by='auc', ascending=False).head(1)['auc'])}")