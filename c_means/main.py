from accuracy_function import accuracy
from c_means import CMeans

if __name__ == '__main__':
    CM = CMeans()
    labels, centers = CM.fuzzyCMeansClustering()

    # a, p, r = accuracy(CM.df, labels, CM.class_labels)

    # print("Accuracy = " + str(a))
    # print("Precision = " + str(p))
    # print("Recall = " + str(r))

    CM.plotResult(centers, labels)
    # CM.plotAllData()
