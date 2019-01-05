# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

0 seems to be Saturdays and 1 = Sundays

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas
from IPython.display import display
sns.set(color_codes=True)

# "prior": orders prior to that users most recent order (~3.2m orders)
# "train": training data supplied to participants (~131k orders)
# "test": test data reserved for machine learning competitions (~75k orders)

# Load data
orders = pandas.read_csv("../dataset/orders.csv")
order_products_train = pandas.read_csv("../dataset/order_products__train.csv")
order_products_prior = pandas.read_csv("../dataset/order_products__prior.csv")
products = pandas.read_csv("../dataset/products.csv")

# visualize orders columns
print(orders.columns)

# visualize orders products train columns
# print(order_products_train.columns)
print(' 1: Total orders by user 1 ')
t_orders_user_1 = orders[orders['user_id']==1]
print(len(t_orders_user_1))

# select train data from orders dataset
orders_train = orders[orders['eval_set']=='train']
print(' 2: Total Train orders by user 1 ')
t_train_orders_user_1 = orders_train[orders_train['user_id']==1]
print(len(t_train_orders_user_1))

# load users priors orders info
orders_prior = orders[orders['eval_set']=='prior']
print(' Total Prior orders plus products by user 1 ')
t_prior_orders_products_user_1 = orders_prior[orders_prior['user_id']==1]
print(len(t_prior_orders_products_user_1))
# orders_products_prior = pandas.merge(orders_prior, order_products_prior, on='order_id', how='right')
#print('orders_products_prior')
#print(orders_products_prior.columns)

# merge orders with products data (to have order_id and product_id)
# orders_products_train = pandas.merge(orders_train, order_products_train, on='order_id', how='right')
# print('3: Total Train orders plus orders by user 1 ')
# t_train_orders_products_user_1 = orders_products_train[orders_products_train['user_id']==1]
# print(len(t_train_orders_products_user_1))
#print(orders_products_train.columns)


df1 = pandas.DataFrame([[1, 1, 5], [1, 2, 3]], columns=['u', 'p', 'c'])
df2 = pandas.DataFrame([[1, 1, 4], [1, 2, 5], [1, 3, 3], [2, 5, 3], [2, 2, 3]], columns=['u', 'p', 'o'])

df3 = pandas.merge(df1, df2, on=('u', 'p'), how='right')
df3['c'] = df3['c'].fillna(0)

df4 = pandas.merge(df1, df2, on=('u', 'p'), how='left')
df4['c'] = df4['c'].fillna(0)