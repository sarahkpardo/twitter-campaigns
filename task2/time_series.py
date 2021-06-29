#!/usr/bin/env python
# coding: utf-8

# In[67]:


import datetime
import functools
import itertools
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[68]:


get_ipython().run_line_magic('matplotlib', 'notebook')
pd.options.display.float_format = '{:.2f}'.format


# In[69]:


get_ipython().system('cd ../data; python3 preprocess.py')


# In[72]:


campaigns = ['GRU202012']
ts = datetime.datetime.now()
log_name = '{}_{}'.format(campaigns, ts)

log = []
log.append('Datasets: {}\n'.format(campaigns))

users_combined = pd.read_csv('../data/users_combined.csv',
                             index_col='userid',
                             low_memory=False)
tweets_combined = pd.read_csv('../data/tweets_combined.csv',
                              index_col='tweetid',
                              low_memory=False)

users = users_combined[users_combined['campaign'].isin(campaigns)]
tweets = tweets_combined[tweets_combined['campaign'].isin(campaigns)]

tweets_dtypes = {
    'user_display_name':'string',
    'user_screen_name':'string',
    'user_reported_location':'string',
    'user_profile_description':'string',
    'user_profile_url':'string',
    'account_creation_date':'datetime64',
    'account_language':'string',
    'tweet_language':'string',
    'tweet_text':'string',
    'tweet_time':'datetime64',
    'tweet_client_name':'category',
    'latitude':'category',
    'longitude':'category',
    'campaign':'string',
    'government':'string',
    'file':'string'}
users_dtypes = {
    'user_display_name':'string',
    'user_screen_name':'string',
    'user_reported_location':'string',
    'user_profile_description':'string',
    'user_profile_url':'string',
    'follower_count':'int64',
    'following_count':'int64',
    'account_creation_date':'datetime64',
    'account_language':'string',
    'campaign':'string',
    'government':'string',
    'file':'string'}

tweets = tweets.astype(tweets_dtypes)
users = users.astype(users_dtypes)


# ## Task 2. Time Series Construction
# 
# A time series is a sequence of data points that consists of successive observations over a given interval of time. Make a time series of the tweet publications in terms of their frequency and occurrence (e.g., determine the number of tweets per hour per campaign).

# In[73]:


tweets_series = tweets
tweets_series.index = tweets['tweet_time']
tweets_series = tweets_series.sort_index()


# In[75]:


resampled = tweets_series.resample('1h')
plt.figure()
resampled.size().plot()
plt.show()


# In[76]:


by_user = tweets_series.groupby('userid').resample('1h')
pivot = pd.pivot_table(pd.DataFrame(by_user.size()), 
                       index=['tweet_time'], 
                       columns=['userid'])


# In[77]:


fig = pivot.plot(legend=False)
plt.show()


# In[78]:


by_user_day = tweets_series.groupby('userid').resample('1d')
pivot_day = pd.pivot_table(pd.DataFrame(by_user_day.size()), 
                           index=['tweet_time'], 
                           columns=['userid'])


# In[82]:


pivot.plot(legend=False, label='hourly')
pivot_day.plot(legend=False, label='daily')
fig.show()


# In[ ]:




