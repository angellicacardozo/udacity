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

# :: Calculating freq for morning, evening and night consumption

morning = orders['order_hour_of_day'] <= 10
night = orders['order_hour_of_day'] > 16

morning_orders = orders[morning]
night_orders = orders[night]

total_orders = len(orders)
total_morning_orders = len(morning_orders)
total_nigth_orders = len(night_orders)
total_evening_orders = total_orders - (total_morning_orders + total_nigth_orders)

freq_total_morning_orders = (total_morning_orders * 100)/total_orders
freq_total_evening_orders = (total_evening_orders * 100)/total_orders
freq_total_nigth_orders = (total_nigth_orders * 100)/total_orders

general_user_behavior['freq_total_morning_orders'] = freq_total_morning_orders
general_user_behavior['freq_total_evening_orders'] = freq_total_evening_orders
general_user_behavior['freq_total_nigth_orders'] = freq_total_nigth_orders

# --------------------------------------------------------------------------------------------------------------------------------

# :: Create indicator for weekend

orders['is_weekend'] = np.where(orders['order_dow'] <= 1, 1, 0)
weekend_df = orders[['order_id', 'is_weekend']]

products_preferences = pandas.merge(weekend_df, products_preferences, on='order_id', how='right', validate='one_to_many')


# --------------------------------------------------------------------------------------------------------------------------------

# :: Preference by a product based on days_since_prior (recurrence of purchase)

# --------------------------------------------------------------------------------------------------------------------------------

# 4 : Build Basket set

baskets = pandas.merge(general_user_behavior, products_preferences, on=['user_id'], how='right', validate='one_to_many')
baskets = pandas.merge(product_user_behavior, baskets, on=['user_id', 'product_id'], how='right')
baskets = baskets.drop(['eval_set', 'order_number'], axis=1)

# 4.4 : Test total orders for user 1 
user_1_baskets = baskets[baskets['user_id']==1]
count_user_1_baskets = user_1_baskets.groupby(['order_id'])['order_id'].aggregate('count')

# 4.2 : Verify the existence for NaNs
print("----------------------------- Baskets : Verify the existence for NaNs ----------------------------------")
print(baskets.isna().sum())

# 4.3 : Update NaNs
baskets['count_reorder_by_user'] = baskets['count_reorder_by_user'].fillna(0)
baskets['count_reorder_product_by_user'] = baskets['count_reorder_product_by_user'].fillna(0)

print(baskets.info(memory_usage='deep'))
print(baskets.memory_usage(deep=True))

# 4.4 : Reduce dataframe in order to test
baskets = baskets.iloc[:20000]

# --------------------------------------------------------------------------------------------------------------------------------

# Random Hyperparameter Grid

# 1 : Create a parameter grid to sample
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# --------------------------------------------------------------------------------------------------------------------------------

# 5 : Build Input for the model

input_y = np.asarray(baskets[['reordered']]['reordered'])
input_x = baskets.drop(['add_to_cart_order', 'reordered', 'days_since_prior_order', 'order_id'], axis=1) 

print(input_x.info(memory_usage='deep'))
print(input_x.memory_usage(deep=True))

from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------------------------------------------------------------

# Search for the best hps

# 5.1 : Create train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.2)

random_classifier = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = random_classifier, 
                               param_distributions = random_grid, 
                               n_iter = 100, cv = 3, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = -1)
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

# --------------------------------------------------------------------------------------------------------------------------------

# 6 : Random Forest Classification training

clf_rf = RandomForestClassifier(random_state = 42)
clf_rf.fit(X_train, y_train)

# Use the resulting model to predict over the testing baskets
prediction_result = clf_rf.predict(X_test)
print(' Random Forest Classification ')
# See score
print("Accuracy on test set: {:0.5f}".format(clf_rf.score(X_test, y_test)))

# Report
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction_result))

best_random_rf = rf_random.best_estimator_
prediction_result_best_random_rf = best_random_rf.predict(X_test)
# See score
print("Accuracy on test set for random parameters: {:0.5f}".format(best_random_rf.score(X_test, y_test)))
print(classification_report(y_test, prediction_result_best_random_rf))
# --------------------------------------------------------------------------------------------------------------------------------

# 7 : Naive Bayes training

from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
prediction_clf_nb = clf_nb.predict(X_test)
print(' Naive Bayes ')

# Report
print("Accuracy on test set: {:0.5f}".format(clf_nb.score(X_test, y_test)))
print(classification_report(y_test, prediction_clf_nb))

# --------------------------------------------------------------------------------------------------------------------------------

features = input_x.columns

predict_proba_rf = best_random_rf.predict_proba(X_test[features])[0:10]
ft_importance = list(zip(X_train[features], best_random_rf.feature_importances_))

# --------------------------------------------------------------------------------------------------------------------------------

top_ten_x_test = np.array(X_test[features]['product_id'][0:10])
tested_products = products.loc[products['product_id'].isin(top_ten_x_test)]
probs_df = pandas.DataFrame({'n_recommended': predict_proba_rf[:,0], 'recommended': predict_proba_rf[:,1]})

merged = pandas.merge(tested_products, X_test[0:10], on=['product_id'], how='left', validate='one_to_one')
merged = pandas.concat([merged, probs_df], axis=1)

to_file = merged.drop(['aisle_id', 'department_id', 'user_id', 'count_reorder_product_by_user', 'count_reorder_by_user', 'count_orders_by_user', 'freq_total_morning_orders', 'freq_total_evening_orders', 'freq_total_nigth_orders', 'is_weekend', 'count_general_item_reorder_freq'], axis=1)
to_file.to_csv('final_result_example.csv', encoding='utf-8', index=False)