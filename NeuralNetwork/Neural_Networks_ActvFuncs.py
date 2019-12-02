#!/usr/bin/env python
# coding: utf-8

# # Import Statements

# In[113]:


import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# # Train and Validation Split

# In[114]:


#Pass the percentage of data you need for test like 20 % 
def train_split(df,test_per):
    indices=df.index.tolist()
    test_size=round(len(df)*(test_per/100))
    random.seed(0)
    test_indices=random.sample(population=indices,k=test_size)
    valid_df=df.loc[test_indices]
    train_df=df.drop(test_indices)
    return train_df,valid_df


# # Activation Functions

# In[115]:


def sigmoid(X):
    sigmoid_ans=np.zeros(X.shape,dtype=np.float128)
    for i in range(0,len(X)):
        x=X[i]
        exp_part=np.exp(-x)
        sigmoid_ans[i]=1/(1+exp_part)
    #print("sigmoid_ans-",sigmoid_ans)
    return sigmoid_ans


def tanh(X):
    tanh_ans=np.zeros(X.shape,dtype=np.float128)
    for i in range(0,len(X)):
        x=X[i]
        exp_part=np.exp(-2*x)
        tanh_ans[i]=(2/(1+exp_part))-1
    # print("tanh_ans-",tanh_ans)
    return tanh_ans


def reLU(X):
    reLU_ans=np.zeros(X.shape,dtype=np.float128)
    for i in range(0,len(X)):
        x=X[i]
        for j in range(len(x)):
            if(x[j] < 0):
                reLU_ans[i][j]=0.01*x[j]
            else:
                reLU_ans[i][j]=x[j]
    # print("reLU_ans-",reLU_ans)
    return reLU_ans


# In[116]:


def softmax(X):
    no_rows,no_labels=X.shape
    soft_array=np.zeros(X.shape)
    for j in range(no_rows):
        x=X[j]
        sum=0
        for i in range(len(x)):
            sum+=np.exp(x[i])
        if(sum != 0):
            for i in range(len(x)):
                soft_array[j][i]=np.exp(x[i])/sum
        else:
            print("Error:Sum is zero")
        
    return soft_array


# # Derivatives of Function

# In[117]:


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def derivative_MSE(output,Y):
    return output-Y

def derivative_tanh(x):
    return 1-(np.power(tanh(x),2))

def derivative_reLU(x):
    ans=np.ones(len(x))
    ans= (x > 0) * 1
    # ans[x < 0] = 0.01
    # print("Deri ReLU-",ans)
    return ans


# # Error Functions

# In[118]:


def MSE_error(actual_Y,predicted_Y):
    m=len(actual_Y)
    MSE=0
    for i in range(0,len(actual_Y)):
        actual=actual_Y[i]
        predicted=predicted_Y[i]
        MSE+=pow((actual-predicted),2)  
    MSE/=m

    return MSE


# In[119]:


# def cal_entropy(df,label):
#     if(df[label].nunique()== 1):
#         return 0
#     else:
#         E=0
#         n_rows,_=df.shape
#         classes=df[label].unique()
#         for cla in classes:
#             p=df[label].value_counts()[cla]
#             if(p==0):
#                 continue
#             p/=n_rows
#             E+=-(p*np.log2(p))
#         return E

def entropy(actual_Y,predicted_Y):
    E=0
    for i in range(len(actual_Y)):
        if(predicted_Y[i] == 0):
            continue
        E-=(actual_Y[i]*np.log2(predicted_Y[i]))
    return E


# # Neural Networks

# In[120]:

def forward_tanh(X,W_H,B_H,W_O,B_O):
     #Forward Propagation
    H_in=X.dot(W_H)+B_H
    H_out=tanh(H_in)
    O_in=H_out.dot(W_O)+B_O
    O_out=softmax(O_in)
    return O_out


def NN_tanh(X,Y,label):
    
    #Initialize layers
    no_samples,no_cols=X.shape
    no_labels=10
    
    no_I=no_cols     #Input layer
    no_O=no_labels  #Output layer
    print("Nodes in Input layer-",no_I)
    print("Nodes in Output layer-",no_O)
    no_H=int(math.sqrt(no_I*no_O))  #Hidden layer-1 Hidden layer
    print("Nodes in Hiddden layer-",no_H)
    
    #Initializing weights and bias
    # W_H=np.random.uniform(size=(no_I,no_H))
    # W_O=np.random.uniform(size=(no_H,no_O))
    # B_H=np.random.uniform(size=(1,no_H))
    # B_O=np.random.uniform(size=(1,no_O))
    W_H=np.random.rand(no_I,no_H)
    W_O=np.random.rand(no_H,no_O)
    B_H=np.random.randn(1,no_H)
    B_O=np.random.randn(1,no_O)
    # print("Initial")
    # print("Weights-",W_O,W_H)
    # print("Biases-",B_O,B_H)
    
    #One-hot encode Y
    Y=one_hot_encode(Y,no_labels)

    alpha=0.01
    epoch=100
    error_cost=[]
    
    while(epoch > 0):
        
        #Forward Propagation
        H_in=X.dot(W_H)+B_H
        # print("H_in-",H_in)
        H_out=tanh(H_in)
        #print("H_out-",H_out)
        O_in=H_out.dot(W_O)+B_O
        #O_out=sigmoid(O_in)
        # print("O_in-",O_in)
        O_out=softmax(O_in)
        # print("O_out-",O_out)
        
        #Backward Propagation
        # Error=entropy(Y,O_out)
        # #Error=Y-O_out
        # #Error=MSE_error(Y,O_out)
        # print("Error-",Error)

        ##Phase 1
        dError_dO_in=derivative_MSE(O_out,Y)
        dO_in_dW_O=H_out
        dError_dW_O=np.dot(dO_in_dW_O.T , dError_dO_in )

        dError_dB_O=dError_dO_in

        ##Phase 2
        dO_in_dH_out=W_O
        dError_dH_out=np.dot(dError_dO_in , dO_in_dH_out.T)
        dH_out_dH_in=derivative_tanh(H_in)
        dH_in_dW_H=X
        dError_dW_H=np.dot(dH_in_dW_H.T , dH_out_dH_in * dError_dH_out )

        dError_dB_H=dError_dH_out * dH_out_dH_in

        # slope_O=derivative_sigmoid(O_out)
        # slope_H=derivative_sigmoid(H_out)
        # delta_O=Error*slope_O
        # Error_H=delta_O.dot(W_O.T)
        # delta_H=Error_H*slope_H
        
        #update weights and bias
        # W_O=W_O+alpha*(H_out.T.dot(delta_O))
        # W_H=W_H+alpha*(X.T.dot(delta_H))        
        # B_O+=np.sum(delta_O)*alpha
        # B_H+=np.sum(delta_H)*alpha

        W_H-=alpha*dError_dW_H
        B_H-=alpha*dError_dB_H.sum(axis=0)
        W_O-=alpha*dError_dW_O
        B_O-=alpha*dError_dB_O.sum(axis=0)

        # print("Weights-",W_O,W_H)
        # print("Biases-",B_O,B_H)
        
        if epoch % 200 == 0:
            loss = np.sum(-Y*np.log(O_out))
            print('Loss function value: ', loss)
            error_cost.append(loss)
        
        epoch-=1
    
    print(O_out)
    #print(error_cost)
    return W_H,B_H,W_O,B_O,O_out


def forward_reLU(X,W_H,B_H,W_O,B_O):
     #Forward Propagation
    H_in=X.dot(W_H)+B_H
    H_out=reLU(H_in)
    O_in=H_out.dot(W_O)+B_O
    O_out=softmax(O_in)
    return O_out



def NN_reLU(X,Y,label):
    
    #Initialize layers
    no_samples,no_cols=X.shape
    no_labels=10
    
    no_I=no_cols     #Input layer
    no_O=no_labels  #Output layer
    print("Nodes in Input layer-",no_I)
    print("Nodes in Output layer-",no_O)
    no_H=int(math.sqrt(no_I*no_O))  #Hidden layer-1 Hidden layer
    print("Nodes in Hiddden layer-",no_H)
    
    #Initializing weights and bias
    # W_H=np.random.uniform(size=(no_I,no_H))
    # W_O=np.random.uniform(size=(no_H,no_O))
    # B_H=np.random.uniform(size=(1,no_H))
    # B_O=np.random.uniform(size=(1,no_O))
    W_H=np.random.rand(no_I,no_H)
    W_O=np.random.rand(no_H,no_O)
    B_H=np.random.randn(1,no_H)
    B_O=np.random.randn(1,no_O)
    # print("Initial")
    # print("Weights-",W_O,W_H)
    # print("Biases-",B_O,B_H)
    
    #One-hot encode Y
    Y=one_hot_encode(Y,no_labels)

    alpha=0.01
    epoch=100
    error_cost=[]
    
    while(epoch > 0):
        
        #Forward Propagation
        H_in=X.dot(W_H)+B_H
        # print("H_in-",H_in)
        H_out=reLU(H_in)
        #print("H_out-",H_out)
        O_in=H_out.dot(W_O)+B_O
        #O_out=sigmoid(O_in)
        # print("O_in-",O_in)
        O_out=softmax(O_in)
        # print("O_out-",O_out)
        
        #Backward Propagation
        # Error=entropy(Y,O_out)
        # #Error=Y-O_out
        # #Error=MSE_error(Y,O_out)
        # print("Error-",Error)

        ##Phase 1
        dError_dO_in=derivative_MSE(O_out,Y)
        dO_in_dW_O=H_out
        dError_dW_O=np.dot(dO_in_dW_O.T , dError_dO_in )

        dError_dB_O=dError_dO_in

        ##Phase 2
        dO_in_dH_out=W_O
        dError_dH_out=np.dot(dError_dO_in , dO_in_dH_out.T)
        dH_out_dH_in=derivative_reLU(H_in)
        dH_in_dW_H=X
        dError_dW_H=np.dot(dH_in_dW_H.T , dH_out_dH_in * dError_dH_out )

        dError_dB_H=dError_dH_out * dH_out_dH_in

        # slope_O=derivative_sigmoid(O_out)
        # slope_H=derivative_sigmoid(H_out)
        # delta_O=Error*slope_O
        # Error_H=delta_O.dot(W_O.T)
        # delta_H=Error_H*slope_H
        
        #update weights and bias
        # W_O=W_O+alpha*(H_out.T.dot(delta_O))
        # W_H=W_H+alpha*(X.T.dot(delta_H))        
        # B_O+=np.sum(delta_O)*alpha
        # B_H+=np.sum(delta_H)*alpha

        W_H-=alpha*dError_dW_H
        B_H-=alpha*dError_dB_H.sum(axis=0)
        W_O-=alpha*dError_dW_O
        B_O-=alpha*dError_dB_O.sum(axis=0)

        # print("Weights-",W_O,W_H)
        # print("Biases-",B_O,B_H)
        
        if epoch % 200 == 0:
            loss = np.sum(-Y*np.log(O_out))
            print('Loss function value: ', loss)
            error_cost.append(loss)
        
        epoch-=1
    
    print(O_out)
    #print(error_cost)
    return W_H,B_H,W_O,B_O,O_out



# # Helper Functions

# In[121]:


#Creating X(data.T),Y arrays from df
def to_arrays(df,label):
    no_rows,no_cols=df.shape
    header=list(df.columns)
    data_array=np.ones((no_cols-1,no_rows))
    for i in range(1,no_cols):             #Removing first col-label
            x=df[header[i]].values
            #data_array[i-1]=x
            data_array[i-1]=mean_normalize(x)
    X=data_array.T

    #Y (output) array
    Y=np.array(df[label].values)  
    return X,Y


#Creating only X array -for testing
def to_array(df,label):
    no_rows,no_cols=df.shape
    header=list(df.columns)
    data_array=np.ones((no_cols,no_rows))
    for i in range(0,no_cols):             #Removing first col-label
            x=df[header[i]].values
            #data_array[i]=x
            data_array[i]=mean_normalize(x)
    X=data_array.T

    return X


# In[122]:


def mean_normalize(x):
    x_new=np.zeros(len(x))
    mean=np.mean(x)
    std=np.std(x)
    #print("x-",x)
    #print("mean-",mean)
    #print("std-",std)
    for i in range(0,len(x)):
        if( int(mean) == 0):
            #print("mean is zero")
            x_new[i]=x[i]
        elif( int(std) == 0 ):
            #print("std is zero")
            x_new[i]=(x[i]- int(mean))
        else:
            x_new[i]=(x[i]-mean)/std
        #print("new x-",x_new[i])
    #print("final x_new-",x_new)
    return x_new


# In[123]:


def one_hot_encode(Y,no_labels):
    encoded_Y=np.zeros((len(Y),no_labels))
    for i in range(0,len(Y)):
        encoded_Y[i,Y[i]]=1
    return encoded_Y


# # Main

# In[124]:


def main():

    #np.warnings.filterwarnings('ignore')
    
    #Loading the file
    file="../Apparel/apparel-trainval.csv"
    label='label'
    df=pd.read_csv(file)
   
    df=df[:1000]
    #Train and Validation Split
    train_df,valid_df=train_split(df,20)
    #train_df=df[:100]
    #train_df=df


    #Training
    train_X,train_Y=to_arrays(train_df,label)
    
    print("tanh")
    W_H,B_H,W_O,B_O,output=NN_tanh(train_X,train_Y,label)

    no_rows,no_labels=output.shape
    correct=0
    for i in range(no_rows):
        prediction=np.argmax(output[i])
        actual=train_Y[i]
        if(actual == prediction):
            correct+=1
        print(prediction)
    accuracy=correct/no_rows
    # print("Training Accuracy-",accuracy)

    #Validation
    valid_X,valid_Y=to_arrays(valid_df,label)
    output=forward_tanh(valid_X,W_H,B_H,W_O,B_O)
    correct=0
    for i in range(len(valid_X)):
        prediction=np.argmax(output[i])
        actual=valid_Y[i]
        if(actual == prediction):
            correct+=1
        print(prediction)
    accuracy=correct/no_rows
    print("Validation Accuracy-",accuracy)

    # print("reLU")
    # W_H,B_H,W_O,B_O,output=NN_reLU(train_X,train_Y,label)

    # no_rows,no_labels=output.shape
    # correct=0
    # for i in range(no_rows):
    #     prediction=np.argmax(output[i])
    #     actual=train_Y[i]
    #     if(actual == prediction):
    #         correct+=1
    #     print(prediction)
    # accuracy=correct/no_rows
    # # print("Training Accuracy-",accuracy)

    # #Validation
    # valid_X,valid_Y=to_arrays(valid_df,label)
    # output=forward_reLU(valid_X,W_H,B_H,W_O,B_O)
    # correct=0
    # for i in range(len(valid_X)):
    #     prediction=np.argmax(output[i])
    #     actual=valid_Y[i]
    #     if(actual == prediction):
    #         correct+=1
    #     print(prediction)
    # accuracy=correct/no_rows
    # print("Validation Accuracy-",accuracy)


    return


# In[125]:


if __name__ == "__main__":
    main()

