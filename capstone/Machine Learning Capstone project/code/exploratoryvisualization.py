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
# Purchase analysis

# Total Orders by Day of Week

# How many order do our users made?

# Orders frequency by Day vs Hour


# Load data
df = pandas.read_csv("../dataset/orders.csv")
df = df.fillna(0)

print(df.columns)

# Histogram by order dow

total_orders = df.count()

sunday_set = df[df['order_dow'] == 0]
saturday_set = df[df['order_dow'] == 1]
monday_set = df[df['order_dow'] == 2]
tuesday_set = df[df['order_dow'] == 3]
wednesday_set = df[df['order_dow'] == 4]
thursday_set = df[df['order_dow'] == 5]
friday_set = df[df['order_dow'] == 6]

print(" Total of orders on Sunday %d" % len(sunday_set))
print(" Total of orders on Saturday %d" % len(saturday_set))
print(" Total of orders on Monday %d" % len(monday_set))

# Volume of orders by week day

weekdays = ('Sunday', 'Saturday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
total_orders = [len(sunday_set), len(saturday_set), len(monday_set), len(tuesday_set), len(wednesday_set), len(thursday_set), len(friday_set)]
y_pos = np.arange(len(weekdays))

plt.figure(figsize=(12,8))
plt.bar(y_pos, total_orders, align='center', alpha=0.5)
plt.xticks(y_pos, weekdays)
plt.ylabel('Total orders')
plt.title(' Volume of orders ')

plt.show()

# How does the volume on Sundays looks like by the hour ?
data = sunday_set['order_hour_of_day'].value_counts().sort_index()
data = data.to_frame()
data['hour'] = data.index

x_pos = data['hour']
y_pos = data['order_hour_of_day']

plt.figure(figsize=(12,8))
plt.bar(x_pos, y_pos, align='center', alpha=0.5)
plt.xticks(np.arange(len(x_pos)), x_pos)
plt.ylabel('Total orders')
plt.title(' Sunday : Volume of orders by hour ')

# Is the same for the others days?
data_saturday_set = saturday_set['order_hour_of_day'].value_counts().sort_index()
data_saturday_set = data_saturday_set.to_frame()
data_saturday_set['hour'] = data_saturday_set.index

data_monday_set = monday_set['order_hour_of_day'].value_counts().sort_index()
data_monday_set = data_monday_set.to_frame()
data_monday_set['hour'] = data_monday_set.index

data_tuesday_set = tuesday_set['order_hour_of_day'].value_counts().sort_index()
data_tuesday_set = data_tuesday_set.to_frame()
data_tuesday_set['hour'] = data_tuesday_set.index

data_wednesday_set = wednesday_set['order_hour_of_day'].value_counts().sort_index()
data_wednesday_set = data_wednesday_set.to_frame()
data_wednesday_set['hour'] = data_wednesday_set.index

data_thursday_set = thursday_set['order_hour_of_day'].value_counts().sort_index()
data_thursday_set = data_thursday_set.to_frame()
data_thursday_set['hour'] = data_thursday_set.index

data_friday_set = friday_set['order_hour_of_day'].value_counts().sort_index()
data_friday_set = data_friday_set.to_frame()
data_friday_set['hour'] = data_friday_set.index

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

x_pos = data_saturday_set['hour']
y_pos = data_saturday_set['order_hour_of_day']
axs[0][0].bar(x_pos, y_pos, align='center', alpha=0.5)
axs[0][0].set_title('Saturday')

x_pos = data_monday_set['hour']
y_pos = data_monday_set['order_hour_of_day']
axs[0][1].bar(x_pos, y_pos, align='center', alpha=0.5)
axs[0][1].set_title('Monday')

x_pos = data_tuesday_set['hour']
y_pos = data_tuesday_set['order_hour_of_day']
axs[0][2].bar(x_pos, y_pos, align='center', alpha=0.5)
axs[0][2].set_title('Tuesday')

x_pos = data_wednesday_set['hour']
y_pos = data_wednesday_set['order_hour_of_day']
axs[1][0].bar(x_pos, y_pos, align='center', alpha=0.5)
axs[1][0].set_title('Wednesday')

x_pos = data_thursday_set['hour']
y_pos = data_thursday_set['order_hour_of_day']
axs[1][1].bar(x_pos, y_pos, align='center', alpha=0.5)
axs[1][1].set_title('Thursday')

x_pos = data_friday_set['hour']
y_pos = data_friday_set['order_hour_of_day']
axs[1][2].bar(x_pos, y_pos, align='center', alpha=0.5)
axs[1][2].set_title('Friday')

fig.suptitle('Week Plotting')