import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import calendar
from dateutil.relativedelta import *

import warnings
warnings.filterwarnings('ignore')

def plot_sales():
    df = pd.read_csv('data/customer_trans.csv', index_col='customer_id')
    
    x = df.customer_id.astype(str).values
    y = df['count'].values
    
    plt.figure(figsize=(20, 10))
    sns.barplot(y, x)
    plt.show()

def plot_sale_frequency_by_mouth(product_id: str):
    df = pd.read_csv('data/2018_product_per_month.csv', index_col='product_id')

    assert product_id in df.index.values, 'product_id does not exist.'

    counts_per_months = df.loc[product_id]

    sns.barplot(counts_per_months.values, counts_per_months.index.values)
    plt.title(f'Transaction frequency for product {product_id} in 2018')
    plt.xlabel('number of transactions')
    plt.ylabel('months')
    plt.legend()
    plt.show()
