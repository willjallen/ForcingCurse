'''
30 years of forbes 400 richest people
'''

import pandas as pd

url = 'http://www.forbes.com/ajax/list/data?year={}&uri=forbes-400&type=person'

big_df = []
for x in range(1990,2023):
	# get json as dataframe
	df = pd.read_json(u:= url.format(str(x)))
	# add year and source url to dataframe
	df['year'] = x
	df['source_url'] = u

	big_df.append(df)

df = pd.concat(big_df)

df.to_csv('forbes400.csv', index=False)