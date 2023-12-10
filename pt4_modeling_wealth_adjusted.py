#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from scipy.stats import pareto
from scipy.stats import linregress
# from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
import seaborn as sns
import os
import shutil
from data import FedData, PSIDData
from utils.helper import calculate_percentiles
from constants import PSID_CHOSEN_PERIOD

#%%
#================================================================
# Output Directory Setup
#================================================================
#region
OUTPUT_DIRECTORY = 'out/pt4'

if os.path.exists(OUTPUT_DIRECTORY):
	shutil.rmtree(OUTPUT_DIRECTORY)
 
os.makedirs(OUTPUT_DIRECTORY)

plt_cnt = 0

def save_fig(plt, name):
	global plt_cnt
	plt.savefig(f'{OUTPUT_DIRECTORY}/{plt_cnt}_{name}')
	plt_cnt += 1
#endregion
#%%
#================================================================
# Plot notator wrapper function
#================================================================
#region
def notate_plot(plt: plt, data_source="simba.isr.umich.edu", website="wallen.me/projects/modeling-wealth", note=""):
	# Adjust the bottom margin to make space for the note
	plt.subplots_adjust(bottom=0.18)

	extra_note = f"Note: {note}" if note else ""

	# Add the data source and website URL to the plot
	note_text = f"Data Source: {data_source} \n More info: {website}\n{extra_note}"
	plt.text(0.95, 0.04, note_text, 
			 ha='right', va='center', transform=plt.gcf().transFigure, fontsize=9, alpha=0.7)
 
#endregion
#%%
#================================================================
# Graph Styling
#================================================================
#region
# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's blue color palette
sns.set_palette(sns.dark_palette("#69d", reverse=False))  
#endregion
#================================================================
# Importing Data
#================================================================
#region
psid_data = PSIDData()
# Equivalence scale adjusts net worth to individuals
equivalence_scale_adjust = False
psid_data.load(cpi_adjust=False, equivalence_scale_adjust=equivalence_scale_adjust, target_year=2019)
psid_wealth_dict = psid_data.get_household_wealth_data()


HOUSEHOLD = not equivalence_scale_adjust
#endregion
#%%
#================================================================
# Picking out a single period
#================================================================
#region
psid_chosen_period_df: pd.DataFrame = psid_wealth_dict[PSID_CHOSEN_PERIOD]
psid_wealth_chosen_period_df: pd.Series = psid_chosen_period_df['IMP WEALTH W/ EQUITY']
#endregion
#%%
#================================================================
# Scale wealth values to normalize negative values to positive
#================================================================
#region

min_value = np.min(psid_wealth_chosen_period_df)

adjusted_psid_wealth_chosen_period_df = psid_wealth_chosen_period_df.apply(lambda x: (x + -min_value + 1) if x < 0 else x)


#endregion
#%%
#================================================================
# Clamped >0 log-log Net Worth emperical CDF 
#================================================================
#region
arr = adjusted_psid_wealth_chosen_period_df 
m = 1
n = 100_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'net_worth_log_log_cdf_plot inclusion ratio: {inclusion_ratio}%')

# Filter the values we want
filtered_arr = arr[(arr > m) & (arr <= n)]

# Sort the data for the empirical CDF
sorted_data = np.sort(filtered_arr)

# Calculate the empirical CDF values
cdf_values = np.arange(1, len(sorted_data)+1) / len(sorted_data)

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
plt.plot(sorted_data, cdf_values, marker='.', linestyle='none', markersize=5, label='Empirical CDF')

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD} - {"Household" if HOUSEHOLD else "Individual"} Empirical CDF of Net Worth')
plt.ylabel(r'Pr$(X \leq x)$')
plt.xlabel('Net Worth')

# y-axis
# plt.yscale('log')

# x-axis
plt.xticks(rotation=45)
plt.xscale('log')
def currency_formatter(x, pos):
	return "${:,.0f}".format(x)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, note="data clamped to range [1, 100,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_log_log_cdf_plot.png')

#endregion