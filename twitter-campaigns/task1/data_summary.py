#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import functools
import io
import itertools
import os
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
pd.options.display.float_format = '{:.2f}'.format


# # 0. Data Import + Preprocessing
# *Store and transform the data into a format to allow tasks to be expressed as [SQL] queries.*

# In[3]:


get_ipython().system('cd ../data; python3 preprocess.py')


# In[8]:


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


# # 1. Data Summarization
# 
# *For each dataset within the repo, create scripts (you can store and transform the data into a database to allow these tasks to be expressed as SQL queries as well) that
# summarize the dataset in terms of:*
# 
# - *total dataset size*
# - *split between tweets and retweets*
# - *number of unique users involved, duration of campaign*
# - *some metric of connectivity of the involved users, e.g., from the follower-following graph or alike (depending on what is accessible)*
# - *(optional) average interaction received by tweets (in terms of retweets, likes, etc.).*
# 

# ### Total dataset size:

# In[9]:


ans = 'User dataset: {} users, {} attributes\n'.format(users.shape[0], users.shape[1])
log.append(ans)
print(ans)

ans = 'Tweets dataset: {} tweets, {} attributes\n'.format(tweets.shape[0], tweets.shape[1])
log.append(ans)
print(ans)


# In[10]:


# number of accounts identified with the campaign
campaign_accounts = len(users.index)

ans = 'Number of accounts identified with campaign: {}\n'.format(campaign_accounts)
log.append(ans)
print(ans)

# number of tweets from these accounts
campaign_tweets = len(tweets.index)
ans = 'Number of tweets by identified accounts: {}\n'.format(campaign_tweets)
log.append(ans)
print(ans)


# ### Duration of campaign:

# In[11]:


duration = tweets.loc[:]['tweet_time'].max() - tweets.loc[:]['tweet_time'].min()

ans = 'Campaign duration: {}\n'.format(duration)
log.append(ans)
print(ans)


# ### User statistics:

# In[12]:


# query group
user_tweets = tweets.groupby('userid')


# In[13]:


active_index = users.index.isin(tweets['userid'])
active_users = users[active_index]
inactive_users = users[~active_index]


# In[14]:


ans = 'Active campaign accounts: {}'.format(len(active_users))
log.append(ans)
print(ans)

ans = 'Inactive campaign accounts: {}'.format(len(inactive_users))
log.append(ans)
print(ans)

ans = 'Total campaign accounts: {}'.format(len(users.index))
log.append(ans)
print(ans)


# In[408]:


tweets_count = user_tweets.size().sort_values()

fig, ax = plt.subplots()
ax.bar(tweets_count.index, tweets_count)
ax.set_xticks([])
ax.set_xlabel('user')
ax.set_ylabel('tweet count')
plt.show()


# ### Identify top user

# In[134]:


max_user = user_tweets.size().idxmax()
users[users.index == max_user]


# ### Tweets vs retweets:

# In[135]:


tweets['is_retweet'].value_counts()


# In[136]:


group_rt = tweets.groupby('is_retweet')
retweets = group_rt.get_group('True')
original_tweets = group_rt.get_group('False')


# In[137]:


rt_count = retweets.size
ans = 'Count of campaign retweets: {}'.format(rt_count)
log.append(ans)
print(ans)

ot_count = original_tweets.size
ans = 'Count of campaign original tweets: {}'.format(ot_count)
log.append(ans)
print(ans)


# In[138]:


retweets = tweets.groupby(['is_retweet', 'userid'])
size = retweets.size().reset_index()


# In[139]:


pivot = pd.pivot_table(size, index='userid', columns='is_retweet')
pivot = pivot.fillna(0).sort_values((0,'True'),ascending=False)


# In[140]:


width = 0.5  # the width of the bars
x = np.arange(len(pivot.index))

plt.figure()
plt.bar(x - width/2, pivot[(0,'False')], width, label='original tweets')
plt.bar(x + width/2, pivot[(0,'True')], width, label='retweets')
plt.yscale('log')
plt.ylabel('tweet count (log scale)')
plt.xlabel('user (sorted by retweet count)')
plt.xticks([])
plt.show()


# ### Retweeted users

# In[141]:


# find userids of retweets
retweeted_users = tweets.groupby('retweet_userid')


# In[142]:


retweeted_users.size()


# ### Connectivity of the involved users:

# Relationships users can have with tweets:
# - author
# - like
# - quote
# - retweet
# - reply
# 
# Relationships users can have with users:
# - replied to
# - quoted
# - retweeted
# 
# Relationships tweets can have with tweets (tweet x {} tweet y):
# - quotes
# - retweets
# - replies

# In[143]:


# retweets between users identified with the campaign
inside_rt = tweets[tweets['retweet_userid'].isin(users.index)]
inside_rt.groupby(['userid','retweet_userid']).size()


# In[144]:


# replies between users identified with the campaign
inside_replies = tweets[tweets['in_reply_to_userid'].isin(users.index)]
inside_replies.groupby(['userid','in_reply_to_userid']).size()


# ### (optional) average interaction received by tweets (in terms of retweets, likes, etc.)

# In[ ]:





# ### Other:

# #### Outliers:

# In[145]:


tweets[tweets['userid'] == max_user]['account_language']


# In[146]:


tweets[tweets['userid'] == max_user]['tweet_language'].value_counts()


# #### User location, account language, and tweet language

# In[147]:


users['user_reported_location'].unique()


# In[148]:


tweets['account_language'].unique()


# In[149]:


tweets['tweet_language'].value_counts()


# In[150]:


language_neq = tweets[tweets['account_language'] != tweets["tweet_language"]]
len(language_neq)


# In[151]:


lat_present = tweets[tweets["latitude"] == "present"]
len(lat_present)

