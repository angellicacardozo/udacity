#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 22:51:12 2018

@author: angellica
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

# Count orders for user 1

count_user_1_orders = len(orders[orders['user_id']==1])

# 1 : Join orders with products information (train with prior).

orders_products_train = pandas.merge(orders[orders['eval_set']=='train'], order_products_train, on='order_id', how='right')
orders_products_prior = pandas.merge(orders[orders['eval_set']=='prior'], order_products_prior, on='order_id', how='right')
frames = [orders_products_train, orders_products_prior]
orders_products = pandas.concat(frames)

# 1.2 : Test total orders for user 1 

user_1_orders_products = orders_products[orders_products['user_id']==1]
count_user_1_orders_products = user_1_orders_products.groupby(['order_id'])['order_id'].aggregate('count')

# --------------------------------------------------------------------------------------------------------------------------------

# 2 : User Behavior :  Features related with user interest

# 2.1 : Select reordered products
products_reordered = orders_products[orders_products['reordered']==1]

# 2.2 : Count reordered frequency for each user
count_items_on_cart = products_reordered.groupby(['user_id'])['add_to_cart_order'].aggregate('count')
count_items_on_cart = count_items_on_cart.reset_index(name='count_reorder_by_user')

count_items_on_cart_user_1 = count_items_on_cart[count_items_on_cart['user_id']==1]

# 2.3 : Count orders made for each user user
count_orders_by_user = orders.groupby('user_id').size()
count_orders_by_user = count_orders_by_user.reset_index(name='count_orders_by_user')

# 2.4 : Count reordered frequency for each user by product
count_user_reorder_by_product = products_reordered.groupby(by=['user_id','product_id'], as_index=False)['reordered'].aggregate('count')
product_user_behavior = count_user_reorder_by_product.rename(columns={'reordered': 'count_reorder_product_by_user'})

# 2.5 : Combine everything as user behavior
general_user_behavior = pandas.merge(count_items_on_cart, count_orders_by_user, on='user_id', how='right')

# 2.6 : Run the basic test again
user_1_orders_products_on_user_behavior = product_user_behavior[product_user_behavior['user_id']==1]

# 2.7 : Verify the existence for NaNs
print("----------------------------- User Behavior : Verify the existence for NaNs ----------------------------------")
print(product_user_behavior.isna().sum())

# --------------------------------------------------------------------------------------------------------------------------------

# 3 : Product Preferences : Preprocessing products set

# 3.1 : Count general rerder interest by product
count_general_reorder_by_product = orders_products.groupby(by=['product_id'], as_index=False)['reordered'].aggregate('count')
count_general_reorder_by_product = count_general_reorder_by_product.rename(columns={'reordered':'count_general_item_reorder_freq'})

# 3.2 : Combine as product preference
products_preferences = pandas.merge(count_general_reorder_by_product, orders_products, on='product_id', how='right', validate='one_to_many')

# 3.3 : Test total orders for user 1 
user_1_products_preferences = products_preferences[products_preferences['user_id']==1]
count_user_1_products_preferences = user_1_products_preferences.groupby(['order_id'])['order_id'].aggregate('count')

# 3.4 : Verify the existence for NaNs
print("----------------------------- Products Preferences : Verify the existence for NaNs ----------------------------------")
print(products_preferences.isna().sum())

# --------------------------------------------------------------------------------------------------------------------------------

# 4 : Build Basket set

baskets = pandas.merge(general_user_behavior, products_preferences, on=['user_id'], how='right', validate='one_to_many')
baskets = pandas.merge(product_user_behavior, baskets, on=['user_id', 'product_id'], how='right')

# 4.4 : Test total orders for user 1 
user_1_baskets = baskets[baskets['user_id']==1]
count_user_1_baskets = user_1_baskets.groupby(['order_id'])['order_id'].aggregate('count')

# 4.2 : Verify the existence for NaNs
print("----------------------------- Baskets : Verify the existence for NaNs ----------------------------------")
print(baskets.isna().sum())

# 4.3 : Update NaNs
baskets['count_reorder_by_user'] = baskets['count_reorder_by_user'].fillna(0)


# baskets = baskets.drop(['eval_set', 'add_to_cart_order', 'reordered', 'days_since_prior_order', 'order_dow', 'order_hour_of_day', 'order_id', 'order_number']) 
