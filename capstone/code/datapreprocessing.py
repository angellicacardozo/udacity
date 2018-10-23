# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas

names = ["order_dow", "order_hour_of_day", "days_since_prior"]
df = pandas.read_csv("../dataset/orders.csv")

ax = df.hist(column="order_hour_of_day", bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
ax = ax[0]
for x in ax:
    
    # Clean lines
    x.spines["right"].set_visible(False)
    x.spines["top"].set_visible(False)
    x.spines["left"].set_visible(False)
    
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
                  
    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Hour of day", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Orders", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

plt.show()