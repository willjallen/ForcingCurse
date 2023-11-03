import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

import seaborn as sns
import os
import shutil
from io import StringIO
from data import FedData, PSIDData
from utils.helper import calculate_percentiles

#================================================================
# Output Directory Setup
#================================================================

OUTPUT_DIRECTORY = 'out/pt1'

if os.path.exists(OUTPUT_DIRECTORY):
	shutil.rmtree(OUTPUT_DIRECTORY)
 
os.makedirs(OUTPUT_DIRECTORY)

plt_cnt = 0

def save_fig(plt, name):
	global plt_cnt
	plt.savefig(f'{OUTPUT_DIRECTORY}/{plt_cnt}_{name}')
	plt_cnt += 1
 
#================================================================
# Graph Styling
#================================================================

# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's green color palette
sns.set_palette(sns.dark_palette("#69d", reverse=False))  
 
 
#================================================================
# Importing Data
#================================================================

fed_data = FedData()
fed_data.load()
net_worth_df = fed_data.get_net_worth_data()

#================================================================
# Net worth by coarse percentiles over time (stacked graph)
#================================================================

# Plotting the data

# Make a copy
net_worth_df_copy = net_worth_df.copy()

# Change the labels
net_worth_df_copy.columns = fed_data.PERCENTILES_STR_LIST

# Set up figure
plt.figure(figsize=(14,8))

# Plot
net_worth_df_copy.plot.area(ax=plt.gca())

# Title and labels
ax=plt.gca()
ax.legend(title="Percentiles")
plt.title('Distribution of Net Worth 1989-2023')
plt.ylabel('Net Worth (Trillions)')
plt.xlabel('Date')

# y-axis
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'net_worth_over_time_stacked.png')

#================================================================
# Picking out a single period
#================================================================

chosen_period = '2020Q1'
net_worth_chosen_period_df = net_worth_df.loc[chosen_period]
# print(net_worth_chosen_period_df)

#================================================================
# Net Worth by percentile category (bar graphs)
#================================================================

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
for category, (start, end) in fed_data.PERCENTILES.items():
	plt.bar(f"{start}-{end}", net_worth_chosen_period_df[category], label=category)

# Title and labels
plt.title(f'{chosen_period} - Net Worth by Population Percentile')
plt.ylabel('Net Worth (Trillions)')
plt.xlabel('Population Percentile')

# y-axis
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"${x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

# x-axis
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'net_worth_dist_bar.png')

#================================================================
# Calculate normalized (Per capita) wealth
#================================================================

# Define the total population
TOTAL_POPULATION = 333_287_557

# Calculate the number of people in each category
people_in_category = {category: TOTAL_POPULATION * size for category, size in fed_data.POPULATION_SIZES.items()}

# Normalize the wealth by number of people in each category
normalized_wealth = net_worth_chosen_period_df / pd.Series(people_in_category)

#================================================================
# Per capita wealth by percentile category (bar graphs)
#================================================================

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
for category, (start, end) in fed_data.PERCENTILES.items():
	plt.bar(f"{start}-{end}", normalized_wealth[category], label=category)

# Title and labels
plt.title(f'{chosen_period} - Net Worth Per capita by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# y-axis
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

# x-axis
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'net_worth_per_capita_bar.png')

#================================================================
# Scaled Per capita wealth by percentile category (bar graphs)
#================================================================

# Initialize the starting point for the first bar
left = 0
# Store the left edges for each bar
bars_left = []
# Store the widths for each bar
bars_width = []

# Calculate the left edges and widths for the bars
for category, pop_size in fed_data.POPULATION_SIZES.items():
	bars_left.append(left)
	width = pop_size * 100
	bars_width.append(width)
	left += width 

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
palette_colors = sns.color_palette()  # Get the palette colors
last_color = palette_colors[-1]  # Get the last color from the palette
for i, category in enumerate(fed_data.POPULATION_SIZES.keys()):
	if i == len(fed_data.POPULATION_SIZES.keys()) - 1:  # Check if it's the last bar
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge', edgecolor=last_color, linewidth=0.5)
	else:
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge')


# Adjust the bottom margin to make space for the note
plt.subplots_adjust(bottom=0.2)
# Add note about exaggeration
plt.text(0.5, 0.02, 'Note: the linewidth of 99.99-100 percentile is exaggerated by 5x compared to the original width', 
         ha='center', va='center', transform=plt.gcf().transFigure)


# Title and labels
plt.title(f'{chosen_period} - Net Worth Per capita by Population Percentile with Proportional Scaling')
plt.ylabel('Net Worth Per capita (Millions)')
plt.xlabel('Population Percentile')

# y-axis
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

# x-axis
modified_percentiles_str_list = fed_data.PERCENTILES_STR_LIST[:-1]
modified_percentiles_str_list[-1] = '99-99.99\n99.99-100'

# Set the x-ticks to be in the middle of each bar for clarity. Remove the very last label
plt.xticks(ticks=[left + (width/2) for left, width in zip(bars_left, bars_width)][:-1], 
		   labels=modified_percentiles_str_list)
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'scaled_net_worth_per_capita_bar.png')


#================================================================
# Per capita log wealth by percentile category (bar graphs)
#================================================================
# Set up figure
plt.figure(figsize=(14, 8))

# Plot
for category, (start, end) in fed_data.PERCENTILES.items():
	plt.bar(f"{start}-{end}", normalized_wealth[category], label=category)
# plt.plot(fed_data.PERCENTILES_STR_LIST, normalized_wealth.values, color='red')

# Title and labels
plt.title(f'{chosen_period} - log Net Worth Per capita by Population Percentile')
plt.ylabel('Net Worth per Capita')
plt.xlabel('Population Percentile')

# y-axis
plt.yscale('log')  

def currency_formatter(x, pos):
	return "${:,.0f}".format(x)

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

# x-axis
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'log_net_worth_per_capita_line+bar.png')

#================================================================
# Scaled Per capita log wealth by percentile category (bar graphs)
#================================================================

# Initialize the starting point for the first bar
left = 0
# Store the left edges for each bar
bars_left = []
# Store the widths for each bar
bars_width = []

# Calculate the left edges and widths for the bars
for category, pop_size in fed_data.POPULATION_SIZES.items():
	bars_left.append(left)
	width = pop_size * 100
	bars_width.append(width)
	left += width 

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
palette_colors = sns.color_palette()  # Get the palette colors
last_color = palette_colors[-1]  # Get the last color from the palette
for i, category in enumerate(fed_data.POPULATION_SIZES.keys()):
	if i == len(fed_data.POPULATION_SIZES.keys()) - 1:  # Check if it's the last bar
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge', edgecolor=last_color, linewidth=0.5)
	else:
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge')

# Adjust the bottom margin to make space for the note
plt.subplots_adjust(bottom=0.2)
# Add note about exaggeration
plt.text(0.5, 0.02, 'Note: the linewidth of 99.99-100 percentile is exaggerated by 5x compared to the original width', 
         ha='center', va='center', transform=plt.gcf().transFigure)

# Title and labels
plt.title(f'{chosen_period} - log Net Worth Per capita by Population Percentile with Proportional Scaling')
plt.ylabel('Net Worth per Capita')
plt.xlabel('Population Percentile')

# y-axis
plt.yscale('log')  

def currency_formatter(x, pos):
	return "${:,.0f}".format(x)

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

# x-axis
modified_percentiles_str_list = fed_data.PERCENTILES_STR_LIST[:-1]
modified_percentiles_str_list[-1] = '99-99.99\n99.99-100'

# Set the x-ticks to be in the middle of each bar for clarity. Remove the very last label
plt.xticks(ticks=[left + (width/2) for left, width in zip(bars_left, bars_width)][:-1], 
		   labels=modified_percentiles_str_list)
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'scaled_log_net_worth_per_capita_bar.png')


#================================================================
# Interpolated Per capita wealth by percentile (bar graphs)
#================================================================

# Extract values into new df
values_df = pd.DataFrame({'values': normalized_wealth.values})

# Calculate the percentiles
# weibull for heavy-tailed distribution
normalized_wealth_percentiles = calculate_percentiles(values_df, 'values', 1, interpolation='linear')

# Increase number of colors
sns.set_palette(sns.dark_palette("#69d", n_colors=101, reverse=False))

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
for percentile, net_wealth in normalized_wealth_percentiles.items():
	plt.bar(x=percentile, height=net_wealth, width=1, label=percentile, align='edge')

# Title and labels
plt.title(f'{chosen_period} - Interpolated Net Worth Per capita by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# y-axis
locs, labels = plt.yticks()
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])

# x-axis
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'interp_net_worth_per_capita_bar.png')

#================================================================
# Interpolated Per capita log wealth by percentile (bar graphs)
#================================================================

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
plt.figure(figsize=(14, 8))
for percentile, net_wealth in normalized_wealth_percentiles.items():
	plt.bar(x=percentile, height=net_wealth, width=1, label=percentile, align='edge')

# Title and labels
plt.title(f'{chosen_period} - Interpolated log Net Worth Per capita by Population Percentile')
plt.ylabel('Net Worth per Capita')
plt.xlabel('Population Percentile')

# y-axis
plt.yscale('log')  

def currency_formatter(x, pos):
	return "${:,.0f}".format(x)

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

# x-axis
plt.xticks(rotation=45)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
save_fig(plt, 'interp_log_net_worth_per_capita_bar.png')



