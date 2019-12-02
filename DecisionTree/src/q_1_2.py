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
    split_val=None
    for val in values:
        temp_df=df[column_names][df[attribute] == val]
        E=cal_entropy(temp_df,label)
        if(E<min_entropy):
            min_entropy=E
            split_val=val
    return split_val
    




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
        if(len(df)==0):
            return tree
        gains=cal_gain(df,label,E_S)
        chosen=maxval(gains)
        if(gains[chosen]== 0.0):
            ans=when_not_pure(df,label)
            return ans

        no_ans=when_not_pure(df,label)

        column_names=df.columns.tolist()
        tree = {chosen:[no_ans,{}]}
        if( chosen in categoric_columns):
            classes=df[chosen].unique()
            for cla in classes:
                temp_df=df[column_names][df[chosen] == cla]
                temp_tree=build_tree(temp_df,label,tree)
                tree[chosen][1][cla]=temp_tree
        elif( chosen in numeric_columns):
            split_val=cal_split_num(df,label,chosen)
            if(split_val):
                temp_df=df[column_names][df[chosen] <= split_val]
                temp_tree=build_tree(temp_df,label,tree)
                query=str('<='+str(split_val))
                tree[chosen][1][query]=temp_tree
                temp_df=df[column_names][df[chosen] > split_val]
                temp_tree=build_tree(temp_df,label,tree)
                query=str('> '+str(split_val))
                tree[chosen][1][query]=temp_tree
        return tree


def predict(inst,tree):
    prediction=None
    try:
        for key in tree.keys():        
            value=inst[key]
            if(key in categoric_columns):
                tree = tree[key][1][value]
            elif(key in numeric_columns):
                num_values=list(tree[key][1].keys())
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
                        tree=tree[key][1]['<='+str(fvalue)]
                    elif(value > fvalue):
                        tree=tree[key][1]['> '+str(fvalue)]
                else:
                    break
            else:
                tree = tree[key][1][value]  

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
# Columns:
# satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,left	promotion_last_5years,sales,salary

def main():
    csv_file="../input/train.csv"
    label='left'
    test_per=20
    
    df=pd.read_csv(csv_file)
    train_df,valid_df=train_split(df,test_per)

    tree=build_tree(train_df,label,None)
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