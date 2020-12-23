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

def plot_top_5(year: int, month: int):
    df = pd.read_csv('data/product_per_month.csv', index_col='product_id')

    selected_date = datetime.date(year, month, 1)
    end_date = str(selected_date)
    start_date = str(selected_date + relativedelta(months=-6))

    top_5_df = df.transpose().loc[start_date:end_date].sum(axis=0).sort_values(ascending=False).head(5)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.bar(top_5_df.index.values, top_5_df.values)
    plt.xlabel('products')
    plt.ylabel('frequency')
    plt.title('top 5 products over last 6 month')

    plt.subplot(2, 1, 2)
    for i in range(50):
        plt.plot(df.transpose().iloc[:, i])

    plt.xlabel('month')
    plt.ylabel('sale frequency')
    plt.title('sale frequency of products per months')
    plt.xticks(rotation=90)

    plt.show()