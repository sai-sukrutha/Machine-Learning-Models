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


########################################
#Using Entropy as Impurity Measure

#Entropy Calculation

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


#Information Gain Calculation

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

        no_ans=when_not_pure(df,label)

        column_names=df.columns.tolist()
        tree = {chosen:[no_ans,{}]}
        if( chosen in categoric_columns):
            classes=df[chosen].unique()
            for cla in classes:
                temp_df=df[column_names][df[chosen] == cla]
                temp_tree=build_tree_entropy(temp_df,label,tree)
                tree[chosen][1][cla]=temp_tree
        elif( chosen in numeric_columns):
            split_val=cal_split_num(df,label,chosen)
            if(split_val):
                temp_df=df[column_names][df[chosen] <= split_val]
                temp_tree=build_tree_entropy(temp_df,label,tree)
                query=str('<='+str(split_val))
                tree[chosen][1][query]=temp_tree
                temp_df=df[column_names][df[chosen] > split_val]
                temp_tree=build_tree_entropy(temp_df,label,tree)
                query=str('> '+str(split_val))
                tree[chosen][1][query]=temp_tree
        return tree

    
########################################
#Using Gini as Impurity Measure


def cal_gini(df,label):
    if(df[label].nunique()== 1):
        return 0
    else:
        G=1
        n_rows,_=df.shape
        if(len(df[label].value_counts())):
            p=(df[label].value_counts()[0])/n_rows
            G=2*p*(1-p)
        return G

def cal_gini_all(df,label,Gstart):
    rows,_=df.shape
    column_names=df.columns.tolist()
    length=len(column_names)
    gini={}
    for i in range(0,length):
        attribute=column_names[i]
        if(attribute==label):
            continue
        if( attribute in categoric_columns):
            classes=df[attribute].unique()
            count=df[attribute].value_counts()
            Gsplit=0
            for cla in classes:
                c=count[cla]
                temp_df=df[column_names][df[attribute] == cla]
                G=cal_gini(temp_df,label)
                Gsplit+=((c/rows)*G)
            gini[attribute]=Gstart-Gsplit
        elif( attribute in numeric_columns):
            Gsplit=0
            value=cal_split_num_gini(df,label,attribute)
            df1=df[column_names][df[attribute] <= value]
            G=cal_gini(df1,label)
            Gsplit+=((len(df1)/rows)*G)
            df3=df[column_names][df[attribute] > value]
            G=cal_gini(df3,label)
            Gsplit+=((len(df3)/rows)*G)
            gini[attribute]=Gstart-Gsplit
    return gini


def cal_split_num_gini(df,label,attribute):
    values=df[attribute].unique()
    column_names=df.columns.tolist()
    max_gini=0
    split_val=values[0]
    for val in values:
        temp_df=df[column_names][df[attribute] == val]
        G=cal_gini(temp_df,label)
        if(G>=max_gini):
            max_gini=G
            split_val=val
    return split_val


def build_tree_gini(df,label,tree):
    #initialization
    if tree is None:                    
        tree={}
    #base condition
    if(is_pure(df,label)):
        ans=when_pure(df,label)
        return ans
    #Recursion
    else:
        Gstart=cal_gini(df,label)
        if(len(df)==0):
            return tree
        gini=cal_gini_all(df,label,Gstart)
        chosen=maxval(gini)
        if(gini[chosen]== 0.0):
            ans=when_not_pure(df,label)
            return ans

        no_ans=when_not_pure(df,label)

        column_names=df.columns.tolist()
        tree = {chosen:[no_ans,{}]}
        if( chosen in categoric_columns):
            classes=df[chosen].unique()
            for cla in classes:
                temp_df=df[column_names][df[chosen] == cla]
                temp_tree=build_tree_gini(temp_df,label,tree)
                tree[chosen][1][cla]=temp_tree
        elif( chosen in numeric_columns):
            split_val=cal_split_num_gini(df,label,chosen)
            if(split_val):
                temp_df=df[column_names][df[chosen] <= split_val]
                temp_tree=build_tree_gini(temp_df,label,tree)
                query=str('<='+str(split_val))
                tree[chosen][1][query]=temp_tree
                temp_df=df[column_names][df[chosen] > split_val]
                temp_tree=build_tree_gini(temp_df,label,tree)
                query=str('> '+str(split_val))
                tree[chosen][1][query]=temp_tree
        return tree


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
    if(TP == 0):
        precision=0
        recall=0
    else:
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
    measures.append(precision)
    measures.append(recall)
    f1score=2/((1/precision)+(1/recall))
    measures.append(f1score)
    return measures



def main():
    csv_file="../input/train.csv"
    label='left'
    test_per=20
    
    df=pd.read_csv(csv_file)
    train_df,valid_df=train_split(df,test_per)

    #Using Entropy
    tree_E=build_tree_entropy(train_df,label,None)
    print("Using Entropy")
    print("-------------------------")
    #pprint.pprint(tree_E)
    measures=confusion_mat(valid_df,tree_E,label)
    print("Accuracy-",measures[0])
    #print("Misclassification-",measures[1])
    print("Precision-",measures[2])
    print("Recall-",measures[3])
    print("F1 score-",measures[4])

    #Using Gini
    tree_G=build_tree_gini(train_df,label,None)
    print("Using Gini")
    print("--------------------------")
    #pprint.pprint(tree_G)
    measures=confusion_mat(valid_df,tree_G,label)
    print("Accuracy-",measures[0])
    #print("Misclassification-",measures[1])
    print("Precision-",measures[2])
    print("Recall-",measures[3])
    print("F1 score-",measures[4])

    return


if __name__ == "__main__":
    main()