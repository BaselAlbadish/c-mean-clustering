def accuracy(df, cluster_labels, class_labels):
    tp = [0, 0]
    tn = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    # ToDo....................................................................**
    for i in range(len(df)):
        if cluster_labels[i] == 1 and class_labels[i] == 'Yes':
            tp[0] = tp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'No':
            tn[0] = tn[0] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'No':
            fp[0] = fp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'Yes':
            fn[0] = fn[0] + 1

    for i in range(len(df)):
        if cluster_labels[i] == 0 and class_labels[i] == 'Yes':
            tp[1] = tp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'No':
            tn[1] = tn[1] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'No':
            fp[1] = fp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'Yes':
            fn[1] = fn[1] + 1

    a0 = float((tp[0] + tn[0])) / (tp[0] + tn[0] + fn[0] + fp[0])
    a1 = float((tp[1] + tn[1])) / (tp[1] + tn[1] + fn[1] + fp[1])
    p0 = float(tp[0]) / (tp[0] + fp[0])
    p1 = float(tp[1]) / (tp[1] + fp[1])
    r0 = float(tp[0]) / (tp[0] + fn[0])
    r1 = float(tp[1]) / (tp[1] + fn[1])

    accuracy = [a0 * 100, a1 * 100]
    precision = [p0 * 100, p1 * 100]
    recall = [r0 * 100, r1 * 100]

    return accuracy, precision, recall
