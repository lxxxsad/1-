# Acc.py
import numpy as np
def acc(L1, L2):
    sum = np.sum(L1[:]==L2[:])
    return sum/len(L2)
# datadvi.py
from scipy.io import loadmat
import numpy as np

def divdata():
    filename = 'C:\\Users\\11696\\Desktop\\2023055987-李笑显\\2024春季机器学习大作业\\数据集' + input("input name of data file: ")
    data = loadmat(filename)
    if filename == 'C:\\Users\\11696\\Desktop\\2023055987-李笑显\\2024春季机器学习大作业\\数据集x1.mat':
        dataX = data[:]
        dataY = data[:][0]
    else:
        dataX = data[:]
        dataY = data[:].T[0]
    print(len(dataX[0]))
    divideornot = input("divide data or not?(Yes/No): ")
    if divideornot == 'Yes':
        dataX_train = []
        dataX_predict = []
        dataY_train = []
        dataY_predict = []
        num_Y = np.unique(dataY).astype(int)
        for i in range(len(num_Y)):
            temp = dataY == num_Y[i]
            temp.astype(float)
            num_Y[i] = np.sum(temp)
            flag = 0
            for j in range(len(dataY)):
                if temp[j] == 1:
                    if flag < int(round(0.9 * num_Y[i])):
                        dataX_train.append(dataX[j])
                        dataY_train.append(dataY[j])
                        flag += 1
                    else:
                        dataX_predict.append(dataX[j])
                        dataY_predict.append(dataY[j])
        dataX_train = np.array(dataX_train)
        dataX_predict = np.array(dataX_predict)
        dataY_train = np.array(dataY_train)
        dataY_predict = np.array(dataY_predict)
        return dataX_train,dataX_predict,dataY_train,dataY_predict
    else:
        return dataX,dataX,dataY,dataY
def decreaseData(dataX,dataY):
    dataX_train = []
    dataY_train = []
    num_Y = np.unique(dataY).astype(int)
    print("this data has {} samples".format(len(dataX)))
    ratio = float(input("input the ratio you want to decrease: "))
    for i in range(len(num_Y)):
        temp = dataY == num_Y[i]
        temp.astype(float)
        num_Y[i] = np.sum(temp)
        flag = 0
        for j in range(len(dataY)):
            if temp[j] == 1:
                if flag < round(ratio * num_Y[i]):
                    dataX_train.append(dataX[j])
                    dataY_train.append(dataY[j])
                    flag += 1
    dataX_train = np.array(dataX_train)
    dataY_train = np.array(dataY_train)
    print(dataX_train)
    return dataX_train,dataY_train
  # kmeans.py
  from sklearn.cluster import KMeans
def k_means(data, clusters):
    return KMeans(n_clusters=clusters,random_state=0).fit(data).predict(data)
  # maplabels.py
  from munkres import Munkres, print_matrix
import numpy as np

def maplabels(L1, L2):
    L2 = L2+1
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)

    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    index = index+1
    print(-G.T)
    print(index)
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in range(nClass2):
        for j in range(len(L2)):
            if L2[j] == index[i, 0]:
                newL2[j] = index[i, 1]
    return newL2
# NMI.py
from sklearn import metrics
def nmi(L1, L2):
    return metrics.normalized_mutual_info_score(L1, L2)
# 大作业.py
import numpy as np
import math
import heapq
import datadvi
import kmeans
import maplabels
import Acc
import NMI
def matrixGaussianKernel_S(X):
    num_samples = len(X)
    S = np.zeros((num_samples,num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            S[i][j] = math.exp(-math.pow(np.linalg.norm(X[i]-X[j]),2)/1.0)
    return S
def initializeIdeMatrix_Q(num_feature):
    I = np.zeros((num_feature,num_feature))
    for i in range(num_feature):
        I[i][i] = 1
    return I
def calculateMatrix_Ls(S):
    D = np.zeros((S.shape[0],S.shape[0]))
    for i in range(S.shape[0]):
        D[i][i] = np.sum(S[i])
    return D-S
def updateQ(epsilon,W):
    Q = np.zeros((W.shape[0],W.shape[0]))
    for i in range(W.shape[0]):
        Q[i][i] = 1/(2*math.sqrt(np.sum(np.square(W[i]))+epsilon))
    return Q
def objectiveFunc(alpha, beta, epsilon, W, Ls, X):
    medTermValue = 0
    for i in range(W.shape[0]):
        medTermValue += math.sqrt(np.sum(np.square(W[i]))+epsilon)
    medTermValue *= alpha
    lastTerm = np.matmul(np.matmul(np.matmul(np.matmul(W.T,X.T),Ls),X),W)
    valueOfObjFun = math.pow(np.linalg.norm(X-np.matmul(X,W)),2) + medTermValue + beta*np.trace(lastTerm)
    return valueOfObjFun
def iterationUntilConvergence(alpha,beta,epsilon,Ls,Q,X):
    temp_W = np.zeros((X.shape[1],X.shape[1]))
    W = np.matmul(np.matmul(
        np.linalg.inv(np.matmul(
            np.matmul(beta * X.T, Ls), X
        ) + np.matmul(X.T, X) + alpha * Q), X.T), X)
    Q = updateQ(epsilon, W)
    #for i in range(50):
    itetimes = 0
    #while(np.sum(np.abs(temp_W-W)) > 0.01):
    for i in range(50):
        print(objectiveFunc(alpha,beta,epsilon,W,Ls,X))
        #print(np.sum(np.abs(temp_W-W)))
        temp_W = W
        itetimes += 1
        W = np.matmul(np.matmul(
            np.linalg.inv(np.matmul(
                np.matmul(beta*X.T,Ls),X
            )+np.matmul(X.T,X)+alpha*Q),X.T),X)
        Q = updateQ(epsilon, W)
        #print(np.sum(np.abs(temp_W - W)))
        #print(Q)
    print(itetimes)
    return W
def rankBasedW(h,W,X):
    norm_W = np.linalg.norm(W,axis=1)
    index = heapq.nlargest(h,range(len(norm_W)),norm_W.take)
    print(index)
    feature_X = np.zeros((X.shape[0],h))
    for i in range(X.shape[0]):
        feature_X[i] = X[i][index]
    return feature_X
if __name__ == '__main__':
    alpha = float(input("input parameter alpha: "))
    beta = float(input("input parameter beta: "))
    epsilon = float(input("input parameter epsilon: "))
    h = int(input("input number of features h: "))
    X,X_pred,Y,Y_pred = datadvi.divdata()
    Q = initializeIdeMatrix_Q(X.shape[1])
    S = matrixGaussianKernel_S(X)
    Ls = calculateMatrix_Ls(S)
    W = iterationUntilConvergence(alpha,beta,epsilon,Ls,Q,X)
    feature_X = rankBasedW(h,W,X)
    print(feature_X.shape)
    lable_pred = kmeans.k_means(feature_X,len(np.unique(Y)))
    lable_pred = maplabels.maplabels(Y,lable_pred)
    print(Acc.acc(Y,lable_pred))
    print(NMI.nmi(Y,lable_pred))
a = np.array([1,2,3])
b = np.array([4,5,6])
print(math.pow(np.linalg.norm(a),2))
print(np.sum(np.square(a)))
print(heapq.nlargest(2,range(len(a)),a.take))
print(a[[1,2]])



