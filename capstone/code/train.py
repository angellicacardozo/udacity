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
print(order_products_train.columns)

# select train data from orders dataset
orders_train = orders[orders['eval_set']=='train']

# merge orders with products data (to have order_id and product_id)
orders_products_train = pandas.merge(order_products_train, orders_train, on='order_id')
print(orders_products_train.columns)

# load users priors orders info
orders_prior = orders[orders['eval_set']=='prior']
orders_products_prior = pandas.merge(order_products_prior, orders_prior, on='order_id')
print('orders_products_prior')
print(orders_products_prior.columns)

# get total orders by user
count_orders_by_user = orders_prior.groupby('user_id').size()
count_orders_by_user = count_orders_by_user.reset_index(name='total_orders')
print('count_orders_by_user')
print(count_orders_by_user.head(10))

# load products infos
products_prior = pandas.merge(orders_products_prior, products, on='product_id')
print('products_prior : columns')
print(products_prior.columns)

# load products reordered : if product was reordered by the user
products_reordered_prior = products_prior[products_prior['reordered']==1]

count_items_on_cart = products_reordered_prior.groupby(['user_id'])['add_to_cart_order'].aggregate('count')
count_items_on_cart = count_items_on_cart.reset_index(name='count_items_on_cart')
print('count_items_on_cart : columns')
print(count_items_on_cart.columns)

# build User Behavior
user_behavior = pandas.merge(count_items_on_cart, count_orders_by_user, on='user_id')
print(' user_behavior : columns ')
print(user_behavior.columns)

# Build baskets to train
baskets = pandas.merge(orders_products_train, user_behavior)
print(' Basket columns')
print(baskets.columns)