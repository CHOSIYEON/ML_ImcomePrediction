# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

pd.set_option('display.max_columns', 100)

# Extract best score(Silhouette) & best parameter

def plot_Max(data, algorithm, best_parameter, best_score):
    if (algorithm == 'DBSCAN'):
        model = DBSCAN(eps=best_parameter['eps'], min_samples=best_parameter['min_samples']).fit(data)
        labels = model.labels_
    elif(algorithm == 'KMeans'):
        centers = []
        model = KMeans(n_clusters=best_parameter['n_clusters'], max_iter=best_parameter['max_iter']).fit(data)
        labels = model.labels_
        centers = model.cluster_centers_
    elif(algorithm == 'EM'):
        model = mixture.GaussianMixture(n_components=best_parameter['n_components'], max_iter=best_parameter['max_iter'], covariance_type='full').fit(data)
        labels = model.predict(data)

    # number of clusters
    unique_labels = set(labels)

    # mapping colors (0~1)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    # outlier have -1 so apeend outliers color(black) to last
    colors.append([0, 0, 0, 1])

    cvec = [colors[label] for label in labels]

    # plot scatter
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1:], c=cvec)
    if (algorithm == 'DBSCAN'):
        plt.title("DBSCAN with eps = %f, min_sample = %d, silhouette = %f" % (best_parameter['eps'], best_parameter['min_samples'], best_score))
    elif (algorithm == 'KMeans'):
        plt.scatter(x=centers[:, 0], y=centers[:, 1], s=50, marker='D', c='r')
        plt.title("K-Means (n_clusters = %d, max_iter = %d silhouette = %f)" % (best_parameter['n_clusters'], best_parameter['max_iter'], best_score))
    elif (algorithm == "EM"):
        plt.title('EM (n_components = %d, max_iter = %d, silhouette =%f)' % (best_parameter['n_components'], best_parameter['max_iter'], best_score))

    plt.show()


# Run DBSCAN algorithm

def exe_DBSCAN(data):
    best_score = 0
    best_parameter = []
    average_score = 0

    for eps in [0.01, 0.05, 0.1, 0.2]:
        for min_samples in [5, 15, 30, 50, 100]:
            # make DBSCAN algorithm and trainiing data
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

            # Numpy array of all the cluster labels assigned to each data point
            labels = db.labels_

            # number of clusters
            unique_labels = set(labels)

            # for outlier
            if (0 in unique_labels):
                average_score = silhouette_score(data, labels)

            # Find best silhouette score
            if (best_score < average_score):
                best_score = average_score
                best_parameter = {'eps': eps, 'min_samples': min_samples}

            # plt--------------------------------------------------------
            # mapping colors (0~1)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            # outlier have -1 so apeend outliers color(black) to last
            colors.append([0, 0, 0, 1])
            cvec = [colors[label] for label in labels]
            # plot scatter
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1:], c=cvec)

            if (0 in unique_labels):
                plt.title("DBSCAN with eps = %d, min_sample = %d, silhouette = %f" % (eps, min_samples, average_score))
            else:
                plt.title("DBSCAN with eps = %d, min_sample = %d" % (eps, min_samples))
            # -----------------------------------------------------------

    plt.clf()
    print("[DBSCAN]")
    print("Best Score: {0} \nBest Parameter: {1}".format(best_score, best_parameter))

    plot_Max(data, 'DBSCAN', best_parameter, best_score)


# Run K-Means cluster

def exe_KMeans(data):
    best_score = 0
    best_parameter = []
    average_score = 0

    for n_clusters in [2, 3, 4, 5, 6]:
        for max_iter in [50, 100, 200, 300]:

            # make K-Means cluster and training data
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(data)

            # predict the data
            labels = kmeans.predict(data)

            # Evaluation using silhouette ---------------------------------------
            score_samples = silhouette_samples(data, labels)
            average_score = silhouette_score(data, labels)

            if (best_score < average_score):
                best_score = average_score
                best_parameter = {'n_clusters': n_clusters, 'max_iter': max_iter}
            # -------------------------------------------------------------------

            # Mark center point.
            centers = kmeans.cluster_centers_

            # number of clusters
            unique_labels = set(labels)

            # plt -----------------------------------------------------------
            # mapping colors (0~1)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            # outlier have -1 so apeend outliers color(black) to last
            colors.append([0, 0, 0, 1])
            cvec = [colors[label] for label in labels]
            # plot scatter
            plt.scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], c=cvec)
            plt.scatter(x=centers[:, 0], y=centers[:, 1], s=50, marker='D', c='r')
            plt.title(
                "K-Means (n_clusters = %d, max_iter = %d silhouette = %f)" % (n_clusters, max_iter, average_score))
            # ----------------------------------------------------------------

    plt.clf()
    print("\n[KMeans]")
    print("Best Score: {0} \nBest Parameter: {1}".format(best_score, best_parameter))

    plot_Max(data, 'KMeans', best_parameter, best_score)


# Run EM cluster

def exe_EM(data):
    best_score = 0
    best_parameter = []
    average_score = 0

    for n_components in [2, 3, 4, 5, 6]:
        for max_iter in [50, 100, 200, 300]:

            # make EM cluster and training data
            em = mixture.GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type='full').fit(data)

            # predict the data
            labels = em.predict(data)

            # number of clusters
            unique_labels = set(labels)

            # ------------------------ Evaluation using silhouette ------------------------
            score_samples = silhouette_samples(data, labels)
            average_score = silhouette_score(data, labels)

            if (best_score < average_score):
                best_score = average_score
                best_parameter = {'n_components': n_components, 'max_iter': max_iter}
            # -----------------------------------------------------------------------------

            # ----------------------------- plt -------------------------------------
            # mapping colors (0~1)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            # outlier have -1 so apeend outliers color(black) to last
            colors.append([0, 0, 0, 1])
            cvec = [colors[label] for label in labels]
            # plot scatter
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1:], c=cvec)
            plt.title('EM (n_components = %d, max_iter = %d, silhouette =%f)' % (n_components, max_iter, average_score))
            # -----------------------------------------------------------------------

    plt.clf()
    print()
    print("\n[EM]")
    print("Best Score: {0} \nBest Parameter: {1}".format(best_score, best_parameter))

    plot_Max(data, 'EM', best_parameter, best_score)



# read data
indicators_data = pd.read_csv("/Users/chosiyeon/Desktop/2020-2/머신러닝/텀프로젝트/Data/Indicators.csv")
print(indicators_data.head())

# Select indicators that seem to be relevant.
indicators = ['CO2 emissions (kt)', 'Rural population (% of total population)', 'Urban population (% of total)',
              'GDP per capita (current US$)', 'Fossil fuel energy consumption (% of total)','Agriculture, value added (% of GDP)', 'Industry, value added (% of GDP)',
              'Manufacturing, value added (% of GDP)']

# Row Name
indices = list(set(indicators_data['CountryCode']))
# Sort alphabetically.
indices = sorted(indices)

# Naming columns
columns = ['CO2 emissions (kt)', 'Rural population (% of total population)', 'Urban population (% of total)',
           'GDP per capita (current US$)', 'Fossil fuel energy consumption (% of total)','Agriculture, value added (% of GDP)',
           'Industry, value added (% of GDP)','Manufacturing, value added (% of GDP)', 'IncomeGroup']

# Make new DataFrame
newData = pd.DataFrame(index=indices, columns=columns)

newData.head()



"""
IndicatorName, IndicatorCode


CO2 emissions (kt) EN.ATM.CO2E.KT
Rural population (% of total population) SP.RUR.TOTL.ZS
Urban population (% of total) SP.URB.TOTL.IN.ZS
GDP per capita (current US$) NY.GDP.PCAP.CD
Fossil fuel energy consumption (% of total) EG.USE.COMM.FO.ZS
Agriculture, value added (% of GDP) NV.AGR.TOTL.ZS
Industry, value added (% of GDP) NV.IND.TOTL.ZS
Manufacturing, value added (% of GDP) NV.IND.MANF.ZS


"""

# Merging appropriate att
indicators_data = indicators_data[(
        (indicators_data['IndicatorCode'] == 'EN.ATM.CO2E.KT') |
        (indicators_data['IndicatorCode'] == 'SP.RUR.TOTL.ZS') |
        (indicators_data['IndicatorCode'] == 'SP.URB.TOTL.IN.ZS') |
        (indicators_data['IndicatorCode'] == 'NY.GDP.PCAP.CD') |
        (indicators_data['IndicatorCode'] == 'EG.USE.COMM.FO.ZS') |
        (indicators_data['IndicatorCode'] == 'NV.AGR.TOTL.ZS') |
        (indicators_data['IndicatorCode'] == 'NV.IND.TOTL.ZS') |
        (indicators_data['IndicatorCode'] == 'NV.IND.MANF.ZS')
)]

# set indicator year to 2005
indicators_data = indicators_data[indicators_data.Year == 2005]

# Fill the table with matching values.
for i in range(len(indicators_data)):
    newData.loc[[indicators_data.iloc[i]['CountryCode']], [indicators_data.iloc[i]['IndicatorName']]] = indicators_data.iloc[i]['Value']

# Import the incomeGroup from Country.csv.
incomeGroup = pd.read_csv("/Users/chosiyeon/Desktop/2020-2/머신러닝/텀프로젝트/Data/Country.csv")

# Make it a list.
label = list(incomeGroup['IncomeGroup'])

# Change string -> int by LabelEncoder to calculate.
le = LabelEncoder()
le.fit(label)
label_encoded_label = le.transform(label)

# Add incomeGroup to the newdata we created.
for i in range(len(incomeGroup)):
    newData.loc[[incomeGroup.iloc[i]['CountryCode']], ['IncomeGroup']] = label_encoded_label[i]

print("----------------------- New Dataset -----------------------\n")
print(newData)
print(newData.describe())
print()

print("----------------------- Missing value Check -----------------------\n")
print(newData.isnull().sum())
print()


# Since the NAN values become 5, change it to the average value and then round up to make it an int.
for i in range(len(indices)):
    if newData.iloc[i]['IncomeGroup'] == 5:
        newData.iloc[i]['IncomeGroup'] = math.ceil(label_encoded_label.mean())

# Filling the NAN values with the average value.
# If there are NAN values, clustering is not possible.
newData.fillna(newData.mean(), inplace=True)

print("----------------------- After missing value handling -----------------------\n")
print(newData.isnull().sum())
print()


# preprocessing before PCA
newData = MinMaxScaler().fit_transform(newData)
newData = pd.DataFrame(data=newData, columns=columns)

print("----------------------- After MinMaxScaler -----------------------\n")
print(newData)
print()

# Analyzing or Modeling using the main components.
# because there are too many Multidimensional data -> hard to clustering and visulization.
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(newData)
newData = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

print("----------------------- After PCA -----------------------\n")
print(newData)
print()


print("----------------------- Algorithm's best score(Silhouette) & best parameter -----------------------\n")
# Execute Clustering -  DBSCAN , KMeans, EM
exe_DBSCAN(newData)
exe_KMeans(newData)
exe_EM(newData)
