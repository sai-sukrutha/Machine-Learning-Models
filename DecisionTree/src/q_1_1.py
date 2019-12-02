
import pandas as pd
import numpy as np
import pprint
import random

categoric_columns=['Work_accident','promotion_last_5years','sales','salary']
numeric_columns=['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']


# # #Train and Validation Split

#Pass the percentage of data you need for test like 20 % 
def train_split(df,test_per):
    indices=df.index.tolist()
    test_size=round(len(df)*(test_per/100))
    random.seed(0)
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
        classes=df[attribute].unique()
        info_gain=0
        for cla in classes:
            temp_df=df[column_names][df[attribute] == cla]
            E=cal_entropy(temp_df,label)
            info_gain+=((len(temp_df)/rows)*E)
        gains[attribute]=E_S-info_gain
    return gains


# # Helper Functions

def maxval(d):  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]
    
    
def is_pure(df,label):
    if(df[label].nunique() <= 1):
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


# ### Representation of Tree
# subtree={question:[max_count,{yes_subtree,no_subtree}]}

def build_tree(df,label,tree):
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
        gains=cal_gain(df,label,E_S)
        chosen=maxval(gains)
        if(gains[chosen]== 0.0):
            ans=when_not_pure(df,label)
            return ans

        no_ans=when_not_pure(df,label)

        column_names=df.columns.tolist()

        tree = {chosen:[no_ans,{}]}
        classes=df[chosen].unique()
        for cla in classes:
            temp_df=df[column_names][df[chosen] == cla]
            temp_tree=build_tree(temp_df,label,tree)
            tree[chosen][1][cla]=temp_tree
        return tree


def predict(inst,tree):
    try:
        for key in tree.keys():        
            value = inst[key]
            tree = tree[key][1][value]
            prediction = None      
            if type(tree) is dict:
                prediction = predict(inst, tree)
            else:
                prediction = tree
                break;
    except KeyError as error:
        key=list(tree.keys())[-1]
        prediction=tree[key][0]    
    return prediction


def confusion_mat(valid_df,tree,label):
    true_val=1
    false_val=0
    total=len(valid_df)
    TN=0
    TP=0
    FP=0
    FN=0
    for i in range(0,total):
        inst=valid_df.iloc[i]
        actual=inst[label]
        predicted=predict(inst,tree)
        if(actual==false_val and predicted==false_val):
            TN+=1
        if(actual==true_val and predicted==false_val):
            FN+=1
        if(actual==false_val and predicted==true_val):
            FP+=1
        if(actual==true_val and predicted==true_val):
            TP+=1
    #measures=[accuracy,misclassification,precision,recall,f1score]
    measures=[]
    accuracy=(TN+TP)/total
    measures.append(accuracy)
    misclassification=(FN+FP)/total
    measures.append(misclassification)
    precision=TP/(TP+FP)
    measures.append(precision)
    recall=TP/(TP+FN)
    measures.append(recall)
    f1score=2/((1/precision)+(1/recall))
    measures.append(f1score)
    return measures


# For train.csv
# Output=left(1,0)
# 
# Categorical Columns:
# salary,sales,promotion_last_5years,Work_accident,left

def main():
    csv_file="../input/train.csv"
    label='left'
    test_per=20
    
    df=pd.read_csv(csv_file)
    train_df,valid_df=train_split(df,test_per)
    
    #Only Categorical
    train_cat_df=train_df[['Work_accident','left','promotion_last_5years','sales','salary']]
    
    tree=build_tree(train_cat_df,label,None)
    pprint.pprint(tree)

    #measures=[accuracy,misclassification,precision,recall,f1score]
    measures=confusion_mat(valid_df,tree,label)
    print("Accuracy-",measures[0])
    #print("Misclassification-",measures[1])
    print("Precision-",measures[2])
    print("Recall-",measures[3])
    print("F1 score-",measures[4])
   
    return


if __name__ == "__main__":
    main()
