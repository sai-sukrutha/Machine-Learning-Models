#!/usr/bin/env python
# coding: utf-8

# # Import Statements

# In[132]:


import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy
import sklearn as skl


# # PCA

# In[133]:


def PCA(X):
    
    #Covariance Matrix
    C=np.cov(X.T)
    
    #Eigen Values
    eigen_values,eigen_vectors=np.linalg.eig(C)
    sum_eigen_values=sum(eigen_values)
    
    #Sorting in decreasing order of eigen values
    comb_vector=zip(eigen_values,eigen_vectors)
    sorted_vector=sorted(comb_vector,reverse=True)
    
    #Finding k  ( n dimensions to k )
    vector2=[]
    for i  in sorted_vector:
        vector2.append((i[0]/sum_eigen_values)*100)
    CumSum = np.cumsum(vector2)
    print("Vector matric is ")
    print(CumSum)
    print("We are choosing 14 dimensions")
    
    #Based on CumSum we choose 14 vectors 
    vector_matrix=[]
    for i in range(0,14):
        vector_matrix.append(sorted_vector[i][1])
        
    #Eigen vector matrix
    W=vector_matrix@C
    
    #Transforming samples to new space
    X_new=X.dot(W.T)
    
    return X_new


# # Part-2 Kmeans

# In[134]:


def K_means(X,K):
    
    _,dims=X.shape #dimensions
    #Initializing K centroids
    C=np.zeros(shape=(dims,K))
    for i in range(dims):
        C[i]=np.random.randint(0,np.max(X),size=K)
    C=C.T
    # print(C)
    
    #Loop
    labels=np.zeros(len(X))
    prev_C=np.zeros(C.shape)
    dist=distance(C,prev_C,None)
    
    while (dist != 0):
        #Cluster Assignment
        for i in range(len(X)):
            dists=[]
            for j in range(len(C)):
                dists.append(distance(X[i], C[j],None))
            label=np.argmin(dists)
            labels[i]=label
        
        #Updating Centroids
        prev_C=copy.deepcopy(C)   #Storing prev C
        for i in range(K):
            points = [X[j] for j in range(len(X)) if labels[j] == i]
            # print("points-",points)
            for p in range(dims):
                avg_points=0
                for j in range(len(points)):
                    avg_points+=points[j][p]
                if(len(points)):
                    avg_points/=len(points)
                else:
                    avg_points=0
                C[i][p]=avg_points

        dist=distance(C,prev_C,None)
        
    # print("Final Centroids-")
    # print(C)
    print("Final Labels")
    print(labels)

    #Clusters
    # colors = ['r', 'g', 'b', 'y', 'c']
    # fig, ax = plt.subplots()
    # for i in range(K):
    #     points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    #     ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    # ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

    plt.scatter(X[:,0],X[:,1], c=labels, cmap='rainbow')
    plt.show()
        
    return labels


# In[135]:


from sklearn.mixture import GaussianMixture

def GMM(X,K,df):
    gmm=GaussianMixture(n_components=K)
    gmm.fit(X)
    # print(gmm.means_)
    # print(gmm.covariances_)
    
    labels = gmm.predict(X)
    print(labels)
    
    plt.scatter(X[:,0],X[:,1], c=labels, cmap='rainbow')
    plt.show()
    
    #print("Converged Log likelihood value",gmm.lower_bound_)
    print("No of iterations",gmm.n_iter_)

    return labels

# In[136]:


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def Hierarchical(X,K):
    cluster = AgglomerativeClustering(n_clusters=K, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)  
    print(cluster.labels_)
    plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
    plt.show()
    
    return cluster.labels_


# # Helper Functions

# In[137]:


def mean_normalize(X):
    for i in range(0,len(X)):
        x_new=np.ones(len(X[i]))
        mean=np.mean(X[i])
        std=np.std(X[i])
        for j in range(0,len(X[i])):
            x_new[j]=(X[i][j]-mean)/std
        X[i]=x_new
    return X


# In[138]:


from sklearn.preprocessing import StandardScaler

def standardize(X):
    X=StandardScaler().fit_transform(X)
    return X



def cal_purity(K,labels,df):
    Y=np.zeros(len(df))

    for i in range(len(df)):
        x=df.loc[i]
        if( x.iloc[-1] == "normal"):
            Y[i]=0
        if( x.iloc[-1] == "dos"):
            Y[i]=1
        if( x.iloc[-1] == "r2l"):
            Y[i]=2
        if( x.iloc[-1] == "u2r"):
            Y[i]=3
        if( x.iloc[-1] == "probe"):
            Y[i]=4

    print("Y -",Y)

    items={}
    for j in range(K):
        items[j]=[]

    for i in range(len(labels)):
        label=int(labels[i])
        #print(label)
        items[label].append(i)

    #print("items-",items) 

    purity=np.zeros(K)

    for j in range(K):
        print("class ",j)
        counts=np.zeros(K)
        for p in items[j]:
            y=int(Y[p])
            counts[y]=counts[y]+1
        print("counts-",counts)
        label=np.argmax(counts)
        print("label-",label)

        correct=0
        for p in items[j]:
            y=int(Y[p])
            #if(label==j):
            if(y == label):
                correct+=1
        print("correct-",correct)
        print(len(items[j]))
        purity[j]=correct/len(items[j])

    print("Purity-",purity)

    labels = '0', '1', '2', '3' ,'4'
    colors = ['r', 'g', 'b', 'y', 'c']
    #explode = (0.1, 0, 0, 0)  # explode 1st slice
    
    # Plot
    plt.pie(purity, explode=None, labels=labels, colors=colors,autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.show()


    return


# In[139]:


# def distance(data,inst,2):  #Euclidean
#     dist=0
#     sum=0
#     for i in range(0,K):
#         sum+=(data[i]-inst[i])**2
#     dist=math.sqrt(sum)
#     return dist


# In[140]:


def distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# # Main

# In[141]:


def main():
    file="intrusion_detection_rsz/data.csv"
    df_orig=pd.read_csv(file)
    
    #Removing the last column - label
    df = df_orig.iloc[:, :-1]
    
    #normalize the data
    X=np.array(df)
    #X=mean_normalize(X)
    X=standardize(X)
    
    #calculate PCA
    print("PCA")
    X_new=PCA(X)
    
    #Part2 - Kmeans
    print("K means")
    K=5
    labels=K_means(X_new,K)
    cal_purity(K,labels,df_orig)
    
    #Part3 - GMM
    #print("GMM")
    #K=5
    #labels=GMM(X_new,K,df)
    # cal_purity(K,labels,df_orig)

    #Part4 - Hierarchical
    #print("Hierarchical")
    #K=5
    #labels=Hierarchical(X_new,K)
    # cal_purity(K,labels,df_orig)


    ##Part5
    #If we have categorical features,they need to be changed to numerical form
    #PCA can be applied then , but it may not work effectively in reducing the dimensions then
    
    return


# In[ ]:


if __name__ == "__main__":
    main()

