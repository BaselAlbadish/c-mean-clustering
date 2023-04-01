import pandas as pd
import random
import operator
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import norm
from scipy.stats import multivariate_normal


class CMeans:
    def __init__(self):
        self.df = pd.read_csv("crime_data.csv")
        self.df = self.df.drop(['Id'], axis=1)
        self.k = 3
        self.MAX_ITER = 300
        self.n = len(self.df)
        self.m = 1.7
        self.error = 0.005

    def initializeMembershipMatrix(self):
        membership_mat = list()
        for i in range(self.n):
            random_num_list = [random.random() for _ in range(self.k)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]
            flag = temp_list.index(max(temp_list))
            for j in range(0, len(temp_list)):
                if j == flag:
                    temp_list[j] = 1
                else:
                    temp_list[j] = 0

            membership_mat.append(temp_list)
        return membership_mat

    def calculateClusterCenter(self, membership_mat):
        cluster_mem_val = list(zip(*membership_mat))
        cluster_centers = []
        for j in range(self.k):
            x = list(cluster_mem_val[j])
            xRaised = [p ** self.m for p in x]
            denominator = sum(xRaised)
            temp_num = []
            for i in range(self.n):
                data_point = list(self.df.iloc[i])
                prod = [xRaised[i] * val for val in data_point]
                temp_num.append(prod)
            numerator = map(sum, list(zip(*temp_num)))
            center = [z / denominator for z in numerator]
            cluster_centers.append(center)
        return cluster_centers

    def updateMembershipValue(self, membership_mat, cluster_centers):
        p = float(2 / (self.m - 1))
        for i in range(self.n):
            x = list(self.df.iloc[i])
            distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in
                         range(self.k)]
            for j in range(self.k):
                den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(self.k)])
                membership_mat[i][j] = float(1 / den)
        return membership_mat

    def getClusters(self, membership_mat):
        cluster_labels = list()
        for i in range(self.n):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels

    def fuzzyCMeansClustering(self):
        membership_mat = self.initializeMembershipMatrix()
        curr = 0
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        lis1, cent_temp = [], []

        for i in range(0, self.k):
            Z = list(np.random.multivariate_normal(mean, cov))
            Z1 = list(np.random.multivariate_normal(mean, cov))
            lis1 = Z + Z1
            cent_temp.append(lis1)

            while curr < self.MAX_ITER:
                membership_mat_new = membership_mat.copy()
                cluster_centers = self.calculateClusterCenter(membership_mat)
                membership_mat = self.updateMembershipValue(membership_mat, cluster_centers)
                cluster_labels = self.getClusters(membership_mat)
                if norm(np.array(membership_mat) - np.array(membership_mat_new)) < self.error:
                    break
                curr += 1
        return cluster_labels, cluster_centers

    def plotDf(self):
        g = sns.PairGrid(self.df)
        g.map(sns.scatterplot)

    def plotAllData(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        i = 0
        for row in axes:
            for ax in row:
                pair = pairs[i]
                ax.scatter(list(self.df.iloc[:, pair[0]]), list(self.df.iloc[:, pair[1]]), marker='o')
                ax.axis('equal')
                i = i + 1
        plt.tight_layout()
        plt.show()

    def plotResult(self, centers, labels):
        set1 = max(set(labels[0:50]), key=labels[0:50].count)
        set2 = max(set(labels[50:100]), key=labels[50:100].count)
        set3 = max(set(labels[100:]), key=labels[100:].count)

        clus1 = np.array([centers[set1][0], centers[set1][1]])
        clus2 = np.array([centers[set2][0], centers[set2][1]])
        clus3 = np.array([centers[set3][0], centers[set3][1]])

        values = np.array(labels)

        searchval_set1 = set1
        searchval_set2 = set2
        searchval_set3 = set3

        ii_set1 = np.where(values == searchval_set1)[0]
        ii_set2 = np.where(values == searchval_set2)[0]
        ii_set3 = np.where(values == searchval_set3)[0]

        ind_set1 = list(ii_set1)
        ind_set2 = list(ii_set2)
        ind_set3 = list(ii_set3)

        Club_df = self.df.iloc[:, 0:2]

        set1_df = Club_df[Club_df.index.isin(ind_set1)]
        set2_df = Club_df[Club_df.index.isin(ind_set2)]
        set3_df = Club_df[Club_df.index.isin(ind_set3)]

        cov_set1 = np.cov(np.transpose(np.array(set1_df)))
        cov_set2 = np.cov(np.transpose(np.array(set2_df)))
        cov_set3 = np.cov(np.transpose(np.array(set3_df)))

        Club_df = np.array(Club_df)

        x1 = np.linspace(4, 8, 150)
        x2 = np.linspace(1.5, 4.5, 150)
        X, Y = np.meshgrid(x1, x2)

        Z1 = multivariate_normal(clus1, cov_set1)
        Z2 = multivariate_normal(clus2, cov_set2)
        Z3 = multivariate_normal(clus3, cov_set3)

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        plt.figure(figsize=(10, 10))
        plt.scatter(Club_df[:, 0], Club_df[:, 1], marker='o')
        plt.contour(X, Y, Z1.pdf(pos), colors="y", alpha=0.5)
        plt.contour(X, Y, Z2.pdf(pos), colors="r", alpha=0.5)
        plt.contour(X, Y, Z3.pdf(pos), colors="g", alpha=0.5)
        plt.axis('equal')
        plt.xlabel('x-Axis', fontsize=16)
        plt.ylabel('Y-Axis', fontsize=16)
        plt.title('Final Clusters', fontsize=22)
        plt.grid()
        plt.show()
