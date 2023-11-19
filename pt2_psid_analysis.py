import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import shutil
from data import FedData, PSIDData
from utils.helper import calculate_percentiles
from constants import PSID_CHOSEN_PERIOD
#================================================================
# Output Directory Setup
#================================================================

OUTPUT_DIRECTORY = 'out/pt2'

if os.path.exists(OUTPUT_DIRECTORY):
	shutil.rmtree(OUTPUT_DIRECTORY)
 
os.makedirs(OUTPUT_DIRECTORY)

plt_cnt = 0

def save_fig(plt, name):
	global plt_cnt
	plt.savefig(f'{OUTPUT_DIRECTORY}/{plt_cnt}_{name}')
	plt_cnt += 1

#================================================================
# Plot notator wrapper function
#================================================================

def notate_plot(plt: plt, data_source="simba.isr.umich.edu", website="wallen.me/projects/modeling-wealth", note=""):
	# Adjust the bottom margin to make space for the note
	plt.subplots_adjust(bottom=0.18)

	extra_note = f"Note: {note}" if note else ""

	# Add the data source and website URL to the plot
	note_text = f"Data Source: {data_source} \n More info: {website}\n{extra_note}"
	plt.text(0.95, 0.04, note_text, 
			 ha='right', va='center', transform=plt.gcf().transFigure, fontsize=9, alpha=0.7)
 
#================================================================
# Graph Styling
#================================================================

# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's green color palette
sns.set_palette(sns.dark_palette("#2e8b57", reverse=True), n_colors=1)  
 
#================================================================
# Importing Data
#================================================================

psid_data = PSIDData()
# Equivalence scale adjusts net worth to individuals
equivalence_scale_adjust = False
psid_data.load(cpi_adjust=False, equivalence_scale_adjust=equivalence_scale_adjust, target_year=2019)
psid_wealth_dict = psid_data.get_household_wealth_data()

HOUSEHOLD = not equivalence_scale_adjust

#================================================================
# Picking out a single period
#================================================================

psid_wealth_chosen_period_df = psid_wealth_dict[PSID_CHOSEN_PERIOD]

#================================================================
# Calculate percentiles
#================================================================

wealth_percentiles = calculate_percentiles(psid_wealth_chosen_period_df, 'IMP WEALTH W/ EQUITY', 0.01)
percentiles_df = pd.DataFrame(list(wealth_percentiles.items()), columns=['Percentile', 'Wealth'])

#================================================================
# Net Worth Histogram, 200 bins
#================================================================
# Set up figure
plt.figure(figsize=(14, 8))

# Plot
plt.hist(psid_wealth_chosen_period_df['IMP WEALTH W/ EQUITY'], bins=200, histtype='stepfilled', alpha=0.8)

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD} - {"Household" if HOUSEHOLD else "Individual"} Net Worth Histogram, 200 bins')
plt.ylabel('Frequency')
plt.xlabel('Net Worth')

# y-axis


# x-axis
plt.xticks(rotation=45)
locs, labels = plt.xticks()  # Get current y-axis tick locations and labels
plt.xticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions


# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'net_worth_hist.png')

#================================================================
# Net Worth histogram, 200 bins, clamped range [-200_000, 2_000_000]
#================================================================
# Include 99% of samples in our clamped histogram
arr = psid_wealth_chosen_period_df['IMP WEALTH W/ EQUITY'] 
m = -200_000
n = 2_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'net_worth_clamped_hist inclusion ratio: {inclusion_ratio}%')


# Set up figure
plt.figure(figsize=(14, 8))

# Plot
plt.hist(psid_wealth_chosen_period_df['IMP WEALTH W/ EQUITY'], bins=200, range=(-200_000, 2_000_000), histtype='stepfilled', alpha=0.8)

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD} - {"Household" if HOUSEHOLD else "Individual"} Net Worth Histogram, 200 bins')
plt.ylabel('Frequency')
plt.xlabel('Net Worth')

# y-axis
# locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
# plt.yticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

# x-axis
plt.xticks(rotation=45)
locs, labels = plt.xticks()  # Get current y-axis tick locations and labels
plt.xticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions
plt.xlim(-200_000, 2_000_000)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, note="data clamped to range [-200,000, 2,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_hist.png')

#================================================================
# Net Worth histogram, 200 bins, symlog
#================================================================
def symlog1p(x):
	return np.sign(x) * np.log1p(np.abs(x))

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
# Create the histogram
counts, bin_edges, patches = plt.hist(symlog1p(psid_wealth_chosen_period_df['IMP WEALTH W/ EQUITY']), bins=200, histtype='stepfilled', alpha=0.8)
# Choose nice, round numbers for your original data range
nice_numbers = np.array([-1e6, -1e5, -1e4, -1e3, -1e2, -1e1, 0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])

# Apply the symlog1p transformation to these nice numbers
transformed_nice_numbers = symlog1p(nice_numbers)

# Set the ticks on the x-axis to the transformed nice numbers
plt.xticks(transformed_nice_numbers, [f"${x:,.0f}" for x in nice_numbers])

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD} - {"Household" if HOUSEHOLD else "Individual"} Net Worth Histogram, 200 bins')
plt.ylabel('Frequency')
plt.xlabel('Net Worth')

# y-axis

# x-axis
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'symlog_net_worth_hist.png')

#================================================================
# log-log Net Worth CDF 
#================================================================
arr = psid_wealth_chosen_period_df['IMP WEALTH W/ EQUITY'] 
m = -100_000_000
n = 100_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'{inclusion_ratio}%')

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
plt.title(f'{PSID_CHOSEN_PERIOD} - Empirical CDF of {"Household" if HOUSEHOLD else "Individual"} Net Worth')
plt.ylabel(r'Pr$(X \leq x)$')
plt.xlabel('symlog Net Worth')

# y-axis
plt.yscale('log')

# x-axis
plt.xticks(rotation=45)
plt.xscale('symlog')
def currency_formatter(x, pos):
	return "${:,.0f}".format(x)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'net_worth_log_log_cdf_plot.png')

#================================================================
# symlog net worth by percentile (scatter)
#================================================================
# Set up figure
plt.figure(figsize=(14, 8))

# Plot
# plt.scatter(percentiles_df['Percentile'], percentiles_df['Wealth'])
plt.plot(percentiles_df['Percentile'], percentiles_df['Wealth'], marker='.', linestyle='none', markersize=5)

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD} - {"Household" if HOUSEHOLD else "Individual"} Net Worth by Population Percentile')
plt.ylabel('Net Worth')
plt.xlabel('Population Percentile')

# y-axis
plt.yscale('symlog') 
def currency_formatter(x, pos):
	return "${:,.0f}".format(x)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

# x-axis
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'symlog_net_worth_percentile_plot.png')