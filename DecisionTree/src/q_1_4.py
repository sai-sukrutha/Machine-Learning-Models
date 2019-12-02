import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(df,attribute1,attribute2,label):
 
    column_names=df.columns.tolist()
    df_1=df[column_names][df[label]==1]
    df_0=df[column_names][df[label]==0]

    x1=df_1[attribute1]
    x0=df_0[attribute1]

    y1=df_1[attribute2]
    y0=df_0[attribute2]

    area=2*np.pi

    plt.xlabel(attribute1)
    plt.ylabel(attribute2)

    plt.scatter(x0, y0, s=area, c='red', alpha=0.5)
    plt.scatter(x1, y1, s=area, c='blue', alpha=0.5)

    plt.show()
    return

def main():
    csv_file="../input/train.csv"
    label='left'
    attribute1='satisfaction_level'
    attribute2='last_evaluation'
    #attribute2='number_project'
    df=pd.read_csv(csv_file)
    plot(df,attribute1,attribute2,label)
    return

if __name__ == "__main__":
    main()