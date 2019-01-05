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
print(' Total orders by user 1 ')
t_orders_user_1 = orders[orders['user_id']==1]

# select train data from orders dataset
orders_train = orders[orders['eval_set']=='train']

# merge orders with products data (to have order_id and product_id)
orders_products_train = pandas.merge(orders_train, order_products_train, on='order_id', how='right')
print(orders_products_train.columns)

# load users priors orders info
orders_prior = orders[orders['eval_set']=='prior']
orders_products_prior = pandas.merge(orders_prior, order_products_prior, on='order_id', how='right')
print('orders_products_prior')
print(orders_products_prior.columns)

# get total orders by user
count_orders_by_user = orders_prior.groupby('user_id').size()
count_orders_by_user = count_orders_by_user.reset_index(name='total_orders')
print('count_orders_by_user')
print(count_orders_by_user.head(10))

# load products infos
products_prior = pandas.merge(products, orders_products_prior, on='product_id', how='right')
print('products_prior : columns')
print(products_prior.columns)

# load products reordered : if product was reordered by the user
products_reordered_prior = products_prior[products_prior['reordered']==1]

count_items_on_cart = products_reordered_prior.groupby(['user_id'])['add_to_cart_order'].aggregate('count')
count_items_on_cart = count_items_on_cart.reset_index(name='count_items_on_cart')
print('count_items_on_cart : columns')
print(count_items_on_cart.columns)

# find out the general interest by an item
count_general_reorder_by_product = orders_products_train.groupby(by=['product_id'], as_index=False)['reordered'].aggregate('count')
count_general_reorder_by_product = count_general_reorder_by_product.rename(columns={'reordered':'general_item_reorder_freq'})
print('count_general_reorder_by_product : columns')
print(count_general_reorder_by_product.columns)
print(count_general_reorder_by_product.head(10))

# build General User Interest
orders_products_set = pandas.merge(count_general_reorder_by_product, orders_products_train, on='product_id', how='right')
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
user_behavior = pandas.merge(count_items_on_cart, count_orders_by_user, on='user_id', how='right')
print(' user_behavior : columns ')
print(user_behavior.columns)

print(' Before merge ')
print(len(orders_products_set))
orders_products_set = pandas.merge(count_user_reorder_by_product, orders_products_set, on=('product_id', 'user_id'), how='right')
orders_products_set['count_reordered_by_user'] = orders_products_set[ 'count_reordered_by_user'].fillna(0)
print(' orders_products_set (count_user_reorder_by_product) : columns ')
print(orders_products_set.columns)
print(' After merge ')
print(len(orders_products_set))

# Build baskets to train
baskets = pandas.merge(user_behavior, orders_products_set, on='user_id', how='right')
baskets = baskets.drop_duplicates()
print(' Basket columns')
print(baskets.columns)

# Training phase
from sklearn.model_selection import train_test_split

baskets_x = baskets.drop(['eval_set', 'add_to_cart_order', 'reordered', 'days_since_prior_order', 'count_reordered_by_user'], axis=1)
baskets_x['count_items_on_cart'] = baskets_x['count_items_on_cart'].fillna(0)
print('Has NaN ? ')
print(baskets_x.isna().sum())
print(' Basket x : columns ')
print(baskets_x.columns)
print(baskets_x.head(10))

baskets_y = baskets[['reordered']]
print(' Basket y : columns ')
print(baskets_y.columns)
print(baskets_y.head(10))

baskets_y = np.asarray(baskets_y['reordered'])

# extract X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(baskets_x, baskets_y, test_size=0.2)

# Random Forest Classification training
# 1 ) (n_estimators = 10, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 24, max_features= 6, random_state = 42)
classifier.fit(X_train, y_train)

# Use the resulting model to predict over the testing baskets
prediction_result = classifier.predict(X_test)

# See score
print("Accuracy on test set: {:0.5f}".format(classifier.score(X_test, y_test)))

# Report
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction_result))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction_clf = clf.predict(X_test)
print(' Naive Bayes ')
print("Accuracy on test set: {:0.5f}".format(clf.score(X_test, y_test)))
print(classification_report(y_test, prediction_clf))