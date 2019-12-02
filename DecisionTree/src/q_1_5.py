import pandas as pd
import numpy as np
import pprint
import random
import matplotlib.pyplot as plt

categoric_columns=['Work_accident','promotion_last_5years','sales','salary']
numeric_columns=['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']


# # #Train and Validation Split
#Pass the percentage of data you need for test like 20 % 
def train_split(df,test_per):
    indices=df.index.tolist()
    test_size=round(len(df)*(test_per/100))
    #random.seed(0)
    test_indices=random.sample(population=indices,k=test_size)
    valid_df=df.loc[test_indices]
    train_df=df.drop(test_indices)
    return train_df,valid_df


# # Entropy Calculation

def cal_entropy(df,label):
    if(df[label].nunique()== 1):
        return 0
    else:
        E=0
        n_rows,_=df.shape
        classes=df[label].unique()
        for cla in classes:
            p=df[label].value_counts()[cla]
            if(p==0):
                continue
            p/=n_rows
            E+=-(p*np.log2(p))
        return E


# # Information Gain Calculation

def cal_gain(df,label,E_S):
    rows,_=df.shape
    column_names=df.columns.tolist()
    length=len(column_names)
    gains={}
    for i in range(0,length):
        attribute=column_names[i]
        if(attribute==label):
            continue
        if( attribute in categoric_columns):
            classes=df[attribute].unique()
            count=df[attribute].value_counts()
            info_gain=0
            for cla in classes:
                c=count[cla]
                temp_df=df[column_names][df[attribute] == cla]
                E=cal_entropy(temp_df,label)
                info_gain+=((c/rows)*E)
            gains[attribute]=E_S-info_gain
        elif( attribute in numeric_columns):
            info_gain=0
            value=cal_split_num(df,label,attribute)
            df1=df[column_names][df[attribute] <= value]
            E=cal_entropy(df1,label)
            info_gain+=((len(df1)/rows)*E)
            df3=df[column_names][df[attribute] > value]
            E=cal_entropy(df3,label)
            info_gain+=((len(df3)/rows)*E)
            gains[attribute]=E_S-info_gain
    return gains

def cal_split_num(df,label,attribute):
    values=df[attribute].unique()
    column_names=df.columns.tolist()
    min_entropy=1
    split_val=values[0]
    for val in values:
        temp_df=df[column_names][df[attribute] == val]
        E=cal_entropy(temp_df,label)
        if(E<=min_entropy):
            min_entropy=E
            split_val=val
    return split_val


# # Helper Functions

def maxval(d):  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def minval(d):  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(min(v))]  
    
def is_pure(df,label):
    if(df[label].nunique() == 1):
        return True
    else:
        return False
    
def when_pure(df,label):
    classes=df[label].unique()
    return classes[0]

def when_not_pure(df,label):
    classes=df[label].unique()
    max=0
    ans=None
    for cla in classes:
        val=len(df[label] == cla)
        if(val >= max):
            ans=cla
            max=val
    return ans

def counts(df,label):
    #count=[1_count or yes_count,0_count or no_count]
    count=[]
    yes_count=df[label].value_counts()[1]
    count.append(yes_count)
    no_count=df[label].value_counts()[0]
    count.append(no_count)
    return count


# ### Representation of Tree
# subtree={chosen:[1_count,0_count,{yes_subtree,no_subtree}]}
#yes_count,no_count is used to give answer if any feature is not part of tree


def build_tree_entropy(df,label,tree):
    #initialization
    if tree is None:                    
        tree={}
    #base condition
    if(is_pure(df,label)):
        ans=when_pure(df,label)
        return ans
    #Recursion
    else:
        E_S=cal_entropy(df,label)
        if(len(df)==0):
            return tree
        gains=cal_gain(df,label,E_S)
        chosen=maxval(gains)
        if(gains[chosen]== 0.0):
            ans=when_not_pure(df,label)
            return ans

        count=counts(df,label)
        column_names=df.columns.tolist()
        #subtree={chosen:[1_count,0_count,{yes_subtree,no_subtree}]}
        tree = {chosen:[count[0],count[1],{}]}
        if( chosen in categoric_columns):
            classes=df[chosen].unique()
            for cla in classes:
                temp_df=df[column_names][df[chosen] == cla]
                temp_tree=build_tree_entropy(temp_df,label,tree)
                tree[chosen][2][cla]=temp_tree
        elif( chosen in numeric_columns):
            split_val=cal_split_num(df,label,chosen)
            if(split_val):
                temp_df=df[column_names][df[chosen] <= split_val]
                temp_tree=build_tree_entropy(temp_df,label,tree)
                query=str('<='+str(split_val))
                tree[chosen][2][query]=temp_tree
                temp_df=df[column_names][df[chosen] > split_val]
                temp_tree=build_tree_entropy(temp_df,label,tree)
                query=str('> '+str(split_val))
                tree[chosen][2][query]=temp_tree
        return tree


def predict(inst,tree,no_nodes):
    prediction=None
    try:
        for key in tree.keys():        
            nodes=tree[key][0]+tree[key][1]
            if(nodes < no_nodes):
                raise KeyError
            value=inst[key]
            if(key in categoric_columns):
                tree = tree[key][2][value]
            elif(key in numeric_columns):
                num_values=list(tree[key][2].keys())
                if(len(num_values)):
                    num_value=num_values[0]
                    if(isinstance(value,np.float64)):
                        num_value=num_value[2:]
                        fvalue=float(num_value)
                    elif(isinstance(value,float)):
                        num_value=num_value[2:]
                        fvalue=float(num_value)
                    elif(isinstance(value,int)):
                        num_value=num_value[2:]
                        fvalue=int(num_value)
                    elif(isinstance(value,np.int64)):
                        num_value=num_value[2:]
                        fvalue=int(num_value)
                    elif(isinstance(value,str)):
                        fvalue=str(num_value)
                        #not cutting [2:]
                    else:
                        if('.' in value):
                            num_value=num_value[2:]
                            fvalue=float(num_value)
                            value=float(value)
                        else:
                            num_value=num_value[2:]
                            fvalue=int(num_value)
                            value=int(value)
                    if(value <= fvalue):
                        tree=tree[key][2]['<='+str(fvalue)]
                    elif(value > fvalue):
                        tree=tree[key][2]['> '+str(fvalue)]
                else:
                    break
            else:
                tree = tree[key][2][value]
    
            if type(tree) is dict:
                prediction = predict(inst, tree,no_nodes)
            else:
                prediction = tree
                break; 
    except KeyError as error:
        key=list(tree.keys())[-1]
        yes_count=tree[key][0]
        no_count=tree[key][1]
        if(yes_count >= no_count):
            prediction=1
        else:
            prediction=0
  
    return prediction



def validation_error(valid_df,tree,label,no_nodes):
    true_val=1
    false_val=0
    total=len(valid_df)
    TN=0
    TP=0
    for i in range(0,total):
        inst=valid_df.iloc[i]
        actual=inst[label]
        predicted=predict(inst,tree,no_nodes)
        if(actual==false_val and predicted==false_val):
            TN+=1
        if(actual==true_val and predicted==true_val):
            TP+=1
    accuracy=(TN+TP)/total
    valid_error=1-accuracy
    return valid_error


def main():
    csv_file="../input/train.csv"
    label='left'
    test_per=20
    
    df=pd.read_csv(csv_file)
    train_df,valid_df=train_split(df,test_per)

    #Using Entropy
    tree=build_tree_entropy(train_df,label,None)
    #pprint.pprint(tree)

    total_nodes=len(train_df)
    x_values=[]
    y_values=[]
    for i in range(0,total_nodes,100):
        x_values.append(i)
        y_values.append(validation_error(valid_df,tree,label,i))

    area=5*np.pi
    plt.xlabel("Number of Nodes")
    plt.ylabel("Validation Error ")
    plt.scatter(x_values, y_values, s=area, c='blue', alpha=0.5)
    plt.show()

    return


if __name__ == "__main__":
    main()

