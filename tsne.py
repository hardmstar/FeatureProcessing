# coding:utf-8
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

casia_file = 'E:/python/SpeechEmoRecSemiSV-PL/casia/feature_emo.pkl'
iemo_file = 'E:/python/SpeechEmoRecSemiSV-PL/iemocap/feature_emo.pkl'
# casia_file = 'E:/python/FeatureProcessing-master/linear/casia_tca.pkl'
# iemo_file = 'E:/python/FeatureProcessing-master/linear/iemo_tca.pkl'


def read_file(feature_emo_file):
    with open(feature_emo_file, 'rb') as f:
        feature_emos = pickle.load(f)                
    X = []; y=[]
    for feature_emo in feature_emos:
        if len(feature_emo[0]) != 300:
            print(len(feature_emo[0]))
        X.append(feature_emo[0].flatten())
        y.append(feature_emo[1])
    X = np.array(X)
    y = np.array(y)

    return X, y

def normalize(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min + 1e-6)
    print(np.where(np.isnan(data)))
    return data

def plt_text(Xs, ys, name, plt):
    for (X, y) in zip(Xs, ys):
        plt.text(X[0], X[1], name+str(y),
            color=plt.cm.Set1(y/10.),
            fontdict={'weight':'bold', 'size':9})

def main():
    casia_X, casia_y = read_file(casia_file)
    iemo_X, iemo_y = read_file(iemo_file)
    casia_X = normalize(casia_X)
    iemo_X = normalize(iemo_X)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    try:
        casia_X = tsne.fit_transform(casia_X)
        iemo_X = tsne.fit_transform(iemo_X)
    except ValueError:
        casia_X = np.real(casia_X)
        iemo_X = np.real(iemo_X)
        casia_X = tsne.fit_transform(casia_X)
    m = np.max(casia_X) if np.max(casia_X)>np.max(iemo_X) else np.max(iemo_X)
    fig = plt.figure()
    ax = plt.subplot(111)
    plt_text(casia_X, casia_y, 'c', plt)
    plt_text(iemo_X, iemo_y, 'i', plt)
    plt.xticks([])
    plt.yticks([])
    plt.axis([-m, m, -m, m])
    plt.show(fig)

if __name__ =='__main__':
    main()

