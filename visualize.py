import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.mixture import GaussianMixture


def read_features(feature_name):
    # filename = 'snapshots/features.npz'
    npzfile = np.load(feature_name)
    return npzfile['arr_0'], npzfile['arr_1']


def draw_bert_features_pca(x, y):
    pca = PCA(2)
    principal_components = pca.fit_transform(x)
    dataframe = pd.DataFrame(data=y, columns=['digit'])
    principalDf = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataframe], axis=1)
    digits = [0, 1]
    colors = ['red', 'green']
    mean = finalDf.groupby('digit').mean()
    s = plt.rcParams['lines.markersize'] ** 2
    for digit, color in zip(digits, colors):
        indices_to_keep = finalDf['digit'] == digit
        plt.scatter(finalDf.loc[indices_to_keep, 'principal component 1'],
                    finalDf.loc[indices_to_keep, 'principal component 2'],
                    c=color, s=s)
        # plt.text(mean.loc[digit, 'principal component 1'], mean.loc[digit, 'principal component 2'], digit,
        #          fontsize=14)
    plt.title("PCA")
    plt.legend(digits)
    plt.grid()
    plt.show()


def draw_bert_features_tsne(x, y, feature_name):
    tsne = TSNE(2)
    principal_components = tsne.fit_transform(x)
    dataframe = pd.DataFrame(data=y, columns=['digit'])
    principalDf = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataframe], axis=1)
    digits = [0, 1]
    colors = ['red', 'green']
    mean = finalDf.groupby('digit').mean()
    s = plt.rcParams['lines.markersize'] ** 2
    for digit, color in zip(digits, colors):
        indices_to_keep = finalDf['digit'] == digit
        plt.scatter(finalDf.loc[indices_to_keep, 'principal component 1'],
                    finalDf.loc[indices_to_keep, 'principal component 2'],
                    c=color, s=s)
        # plt.text(mean.loc[digit, 'principal component 1'], mean.loc[digit, 'principal component 2'], digit,
        #          fontsize=14)
    plt.title("tSNE")
    plt.legend(digits)
    plt.grid()
    plt.show()


def build_gmm(x, y):
    gmm = GaussianMixture(n_components=4).fit(x, y)
    labels = gmm.predict(x)
    counts = labels == y
    counts_wrong = np.unique(counts, return_counts=True)
    print(counts_wrong)


def main():
    feature_name = 'snapshots/src_books_features.npz'
    x, y = read_features(feature_name)
    # draw_bert_features_tsne(x, y, feature_name)
    # draw_bert_features_pca(x, y)
    build_gmm(x, y)


if __name__ == '__main__':
    main()
