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
    df = pd.read_csv('data/customer_trans.csv')
    
    x = df.customer_id.astype(str).values
    y = df['count'].values
    
    plt.figure(figsize=(20, 10))
    sns.barplot(y, x)
    plt.show()