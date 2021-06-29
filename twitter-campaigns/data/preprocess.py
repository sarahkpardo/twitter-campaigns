import datetime
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


p = Path('users')

users_combined = pd.DataFrame()

for file in p.iterdir():
    df = pd.read_csv(file, index_col='userid', low_memory=False)

    # add columns to identify separate campaigns
    df['campaign'] = file.name.split('_')[-5]+file.name.split('_')[-4]
    df['release'] = file.name.split('_')[-4]
    df['government'] = file.name.split('_')[-5]
    df['file'] = str(file.stem)
    users_combined = users_combined.append(df)

users_combined.info()
users_combined.to_csv('users_combined.csv')


p = Path('tweets')

tweets_combined = pd.DataFrame()

for file in p.iterdir():
    df = pd.read_csv(file, index_col='tweetid', low_memory=False)

    # add columns to identify separate campaigns
    df['campaign'] = file.name.split('_')[-5]+file.name.split('_')[-4]
    df['release'] = file.name.split('_')[-4]
    df['government'] = file.name.split('_')[-5]
    df['file'] = str(file.stem)
    tweets_combined = tweets_combined.append(df)

tweets_combined.info()
tweets_combined.to_csv('tweets_combined.csv')

"""
if not Path('twitter_election_integrity_hashed.db').exists():
    Path('twitter_election_integrity_hashed.db').touch()

con = sqlite3.connect('twitter_election_integrity_hashed.db')
cur = con.cursor()

users_combined.to_sql('users', con, if_exists='append', index=False)
tweets_combined.to_sql('tweets', con, if_exists='append', index=False)

con.commit()
con.close()
"""
