import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

def quanQual(dataset):
    """
    Splits dataset columns into quantitative (numerical) and qualitative (categorical) lists.
    """
    quan = []
    qual = []

    for columnName in dataset.columns:
        if dataset[columnName].dtype == 'object':
            qual.append(columnName)
        else:
            quan.append(columnName)
    return quan,qual

def univaiate(dataset,quan):
    """
    Generates descriptive statistics and outlier thresholds for quantitative columns.
    """
    descriptive = pd.DataFrame(index = ["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","Q4:100%","IQR","1.5Rule","Lesser","Greater","Min","Max","Skew","Kurtosis","Var","Std_Deviation"],columns=quan)
    for columnName in quan:
        descriptive[columnName]["Mean"]=dataset[columnName].mean()
        descriptive[columnName]["Median"]=dataset[columnName].median()
        descriptive[columnName]["Mode"]=dataset[columnName].mode()[0]
        descriptive[columnName]["Q1:25%"]=dataset.describe()[columnName]["25%"]
        descriptive[columnName]["Q2:50%"]=dataset.describe()[columnName]["50%"]
        descriptive[columnName]["Q3:75%"]=dataset.describe()[columnName]["75%"]
        descriptive[columnName]["Q4:100%"]=dataset.describe()[columnName]["max"]
        descriptive[columnName]["IQR"]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
        descriptive[columnName]["1.5Rule"]=1.5*descriptive[columnName]["IQR"]
        descriptive[columnName]["Lesser"]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5Rule"]
        descriptive[columnName]["Greater"]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5Rule"]
        descriptive[columnName]["Min"]=dataset[columnName].min()
        descriptive[columnName]["Max"]=dataset[columnName].max()
        descriptive[columnName]["Skew"] = dataset[columnName].skew()
        descriptive[columnName]["Kurtosis"] = dataset[columnName].kurtosis()
        descriptive[columnName]["Var"] = dataset[columnName].var()
        descriptive[columnName]["Std_Deviation"] = dataset[columnName].std()
    return descriptive

def replace_outlier(dataset, columns, descriptive):
    """
    Replaces outliers in dataset with threshold values from descriptive statistics.
    """
    for columnName in columns:
        lower_bound = descriptive[columnName]["Lesser"]
        upper_bound = descriptive[columnName]["Greater"]

        dataset.loc[dataset[columnName] < lower_bound, columnName] = lower_bound
        dataset.loc[dataset[columnName] > upper_bound, columnName] = upper_bound

    return dataset