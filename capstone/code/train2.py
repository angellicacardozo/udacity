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

# 3 : Features related with user interest

# 3.1 : Select reordered products
products_reordered = orders_products[orders_products['reordered']==1]

# 3.2 : Count reordered frequency for each user
count_items_on_cart = products_reordered.groupby(['user_id'])['add_to_cart_order'].aggregate('count')
count_items_on_cart = count_items_on_cart.reset_index(name='count_reorder_by_user')

count_items_on_cart_user_1 = count_items_on_cart[count_items_on_cart['user_id']==1]

# 3.3 : Count orders made for each user user
count_orders_by_user = orders.groupby('user_id').size()
count_orders_by_user = count_orders_by_user.reset_index(name='count_orders_by_user')

# 3.4 : Count reordered frequency for each user by product
count_user_reorder_by_product = products_reordered.groupby(by=['user_id','product_id'], as_index=False)['reordered'].aggregate('count')
count_user_reorder_by_product = count_user_reorder_by_product.rename(columns={'reordered': 'count_reorder_product_by_user'})

user_1_count_orders_by_user = count_user_reorder_by_product[count_user_reorder_by_product['user_id']==1]

# 3.5 : Combine everything as user behavior
user_behavior = pandas.merge(count_items_on_cart, count_orders_by_user, on='user_id', how='right')
user_behavior = pandas.merge(user_behavior, count_user_reorder_by_product, on='user_id', how='right', validate='one_to_many')

# 3.6 : Run the basic test again
user_1_orders_products_on_user_behavior = user_behavior[user_behavior['user_id']==1]