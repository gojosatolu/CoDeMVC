from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=None):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    
    zs_list = [[] for _ in range(view)]
    rs_list = [[] for _ in range(view)]
    H_list = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (xs, y, _) in enumerate(test_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            
            xrs, zs, rs, H = model(xs)
            
            for v in range(view):
                zs_list[v].append(zs[v].cpu().data.numpy())
                rs_list[v].append(rs[v].cpu().data.numpy())
            H_list.append(H.cpu().data.numpy())
            labels_list.append(y.numpy())

    # Concatenate all batches
    zs = [np.concatenate(z, axis=0) for z in zs_list]
    rs = [np.concatenate(r, axis=0) for r in rs_list]
    H = np.concatenate(H_list, axis=0)
    labels = np.concatenate(labels_list, axis=0).squeeze()

    if eval_h:
        print("Clustering results on low-level features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(zs[v])
            nmi, ari, acc, pur = evaluate(labels, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))
        print("Clustering results on view-consensus features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(rs[v])
            nmi, ari, acc, pur = evaluate(labels, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))

    # Clustering results on global features
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(H)
    nmi, ari, acc, pur = evaluate(labels, y_pred)
    if epoch is not None:
        print('Epoch {}'.format(epoch),'The clustering performace: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    else:
        print('The clustering performace: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur