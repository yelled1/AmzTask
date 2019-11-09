import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from load_data import load_parse_save
from hlper_func import train_validate_test_split

tqdm.pandas(desc='load & parse csv')
#prev_df = load_parse_save(save_file='./parsed_dataframe.pkl', debug=True)
#prev_df.to_csv('/tmp/prev_maint_parsed.csv')

prev_df = pd.read_pickle('./parsed_dataframe.pkl')
#test_g_df = test_df[((test_df.daysDelta.isna()) | (test_df.daysDelta > 0.0))] #.to_csv('/tmp/test_base.csv')
prev_df['Fset'] = prev_df.daysDelta.apply(lambda x: True if x <= 0.0 else False)

#prev_df[prev_df.failure].head()
devices = pd.DataFrame(prev_df.device.value_counts().reset_index())
train_dev, validate_dev, test_dev = train_validate_test_split(devices, seed=312)
fail_set = set(prev_df.device[prev_df.failure].tolist())

# checking to see how many failed set in the each category
print('train', train_dev.shape, len(set(train_dev['index'].tolist()) & fail_set))
print('test', test_dev.shape, len(set(test_dev['index'].tolist()) & fail_set))
print('validate', validate_dev.shape, len(set(validate_dev['index'].tolist()) & fail_set))

test_df = prev_df[prev_df.device.apply(lambda x: x in test_dev['index'].tolist())]
train_df = prev_df[prev_df.device.apply(lambda x: x in train_dev['index'].tolist())]
valid_df = prev_df[prev_df.device.apply(lambda x: x in validate_dev['index'].tolist())]

for i in range(1, 10):
  metric_ix = 'metric%d' %i
  print("{} {:,.3f}:{:,.3f} {:,.3f} | {:,.3f}:{:,.3f} {:,.3f}".format(metric_ix, 
                                    test_df[~test_df.Fset][metric_ix].mean(),
                                    test_df[~test_df.Fset][metric_ix].max(),
                                    test_df[~test_df.Fset][metric_ix].min(),
                                    test_df[test_df.Fset][metric_ix].mean(),
                                    test_df[test_df.Fset][metric_ix].max(),
                                    test_df[test_df.Fset][metric_ix].min(),
                                   ))

ggrp = test_df.groupby(['Fset', 'device'])
ggrp.keys

for g in ggrp.groups: 
  print(g)
