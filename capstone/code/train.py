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
from sklearn.metrics import confusion_matrix
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

# find out the general interest by an item
count_general_reorder_by_product = products_reordered_prior.groupby(['product_id'])['reordered'].aggregate('count')
count_general_reorder_by_product = count_general_reorder_by_product.reset_index(name='general_item_reorder_freq')
print('count_general_reorder_by_product : columns')
print(count_general_reorder_by_product.columns)
print(count_general_reorder_by_product.head(10))

# build General User Interest
orders_products_set = pandas.merge(orders_products_train, count_general_reorder_by_product, on='product_id')
print(' orders_products_set : columns ')
print(orders_products_set.columns)

# build specif interest on item
count_user_reorder_by_product = products_prior.groupby(by=['user_id','product_id'], as_index=False)['reordered'].aggregate('count')
count_user_reorder_by_product = count_user_reorder_by_product.rename(columns={'reordered': 'count_reordered_by_user'})
#count_user_reorder_by_product = count_user_reorder_by_product.to_frame()
#count_user_reorder_by_product = count_user_reorder_by_product.reset_index(name='count_user_reorder_by_product')
print(' count_user_reorder_by_product : columns ')
print(count_user_reorder_by_product.columns)
print(count_user_reorder_by_product.head(10))

# build User Behavior
user_behavior = pandas.merge(count_items_on_cart, count_orders_by_user, on='user_id')
print(' user_behavior : columns ')
print(user_behavior.columns)

print(' Before merge ')
print(len(orders_products_set))
orders_products_set = pandas.merge(orders_products_set, count_user_reorder_by_product, on=('product_id', 'user_id'), how='left')
orders_products_set['count_reordered_by_user'] = orders_products_set[ 'count_reordered_by_user'].fillna(0)
print(' orders_products_set (count_user_reorder_by_product) : columns ')
print(orders_products_set.columns)
print(' After merge ')
print(len(orders_products_set))

# Build baskets to train
baskets = pandas.merge(orders_products_set, user_behavior)
print(' Basket columns')
print(baskets.columns)

# Training phase
from sklearn.model_selection import train_test_split

baskets_x = baskets.drop(['eval_set', 'add_to_cart_order', 'reordered', 'days_since_prior_order'], axis=1)
print(' Basket x : columns ')
print(baskets_x.columns)
print(baskets_x.head(10))

baskets_y = baskets_x[['reordered']]
print(' Basket y : columns ')
print(baskets_y.columns)
print(baskets_y.head(10))

# extract X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(baskets_x, baskets_y, test_size=0.2)

# Random Forest Classification training
# 1 ) (n_estimators = 10, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 24, max_features= 6, random_state = 42)
classifier.fit(X_train,y_train)

# Use the resulting model to predict over the testing baskets
prediction_result = classifier.predict(X_test)

# See score
print("Accuracy on test set: {:0.5f}".format(classifier.score(X_test, y_test)))

# Report
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction_result))

cm = confusion_matrix(y_test ,prediction_result)
print(cm)