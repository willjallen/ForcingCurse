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
# print(net_worth_df)

#================================================================
# Net worth by coarse percentiles over time (stacked graph)
#================================================================

# Plotting the data

# Make a copy
net_worth_df_copy = net_worth_df.copy()

# Change the labels
net_worth_df_copy.columns = fed_data.PERCENTILES_STR_LIST

plt.figure(figsize=(14,8))
ax=plt.gca()
net_worth_df_copy.plot.area(ax=plt.gca())
ax.legend(title="Percentiles")
plt.title('Distribution of Net Worth 1989-2023')
plt.ylabel('Net Worth (Trillions)')
# Adjust the y-axis to be in trillions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions
plt.xlabel('Date')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'net_worth_over_time_stacked.png')

# Reset the color pallete
# sns.set_palette(sns.dark_palette("#69d", reverse=True)) 


#================================================================
# Picking out a single period
#================================================================

chosen_period = '2020Q1'
net_worth_chosen_period_df = net_worth_df.loc[chosen_period]
# print(net_worth_chosen_period_df)

#================================================================
# Net Worth by percentile category (bar graphs)
#================================================================

# Plotting the data
plt.figure(figsize=(14, 8))
for category, (start, end) in fed_data.PERCENTILES.items():
    plt.bar(f"{start}-{end}", net_worth_chosen_period_df[category], label=category)
plt.title(f'Net Worth Distribution in {chosen_period} by Population Percentile')
plt.ylabel('Net Worth (Trillions)')
# Adjust the y-axis to be in trillions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions
plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'net_worth_dist_bar.png')


#================================================================
# Per-capita wealth by percentile category (bar graphs)
#================================================================

# Define the total population
TOTAL_POPULATION = 333_287_557

# Calculate the number of people in each category
people_in_category = {category: TOTAL_POPULATION * size for category, size in fed_data.POPULATION_SIZES.items()}

# Normalize the wealth by number of people in each category
normalized_wealth = net_worth_chosen_period_df / pd.Series(people_in_category)

# Plotting the bar and line graph
plt.figure(figsize=(14, 8))
for category, (start, end) in fed_data.PERCENTILES.items():
    plt.bar(f"{start}-{end}", normalized_wealth[category], label=category)
plt.title(f'Per-capita Wealth Distribution in {chosen_period} by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
# Adjust the y-axis to be in millions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in millions
plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'net_worth_per_capita_bar.png')


#================================================================
# Per-capita wealth by percentile category (bar + line graphs)
#================================================================
plt.figure(figsize=(14, 8))
for category, (start, end) in fed_data.PERCENTILES.items():
    plt.bar(f"{start}-{end}", normalized_wealth[category], label=category)
plt.plot(fed_data.PERCENTILES_STR_LIST, normalized_wealth.values, color='red')
plt.title(f'Per-capita Wealth Distribution in {chosen_period} by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
# Adjust the y-axis to be in millions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in millions
plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'net_worth_per_capita_line+bar.png')

#================================================================
# Per-capita log wealth by percentile category (bar + line graphs)
#================================================================
plt.figure(figsize=(14, 8))
for category, (start, end) in fed_data.PERCENTILES.items():
    plt.bar(f"{start}-{end}", normalized_wealth[category], label=category)
plt.plot(fed_data.PERCENTILES_STR_LIST, normalized_wealth.values, color='red')
plt.title(f'Per-capita log Wealth Distribution in {chosen_period} by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
# Adjust the y-axis to be in millions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in millions
plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
plt.yscale('log')  
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'log_net_worth_per_capita_line+bar.png')

#================================================================
# Scaled Per-capita log wealth by percentile category (bar graphs)
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

# Plot the bars
plt.figure(figsize=(14, 8))
palette_colors = sns.color_palette()  # Get the palette colors
last_color = palette_colors[-1]  # Get the last color from the palette
for i, category in enumerate(fed_data.POPULATION_SIZES.keys()):
	if i == len(fed_data.POPULATION_SIZES.keys()) - 1:  # Check if it's the last bar
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge', edgecolor=last_color, linewidth=1.5)
	else:
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge')

# Add a title and labels
plt.title('Scaled Per-capita log Wealth Distribution by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])

modified_percentiles_str_list = fed_data.PERCENTILES_STR_LIST[:-1]
modified_percentiles_str_list[-1] = '99-99.99\n99.99-100'

# Set the x-ticks to be in the middle of each bar for clarity. Remove the very last label
plt.xticks(ticks=[left + (width/2) for left, width in zip(bars_left, bars_width)][:-1], 
           labels=modified_percentiles_str_list)
plt.yscale('log')  

plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'scaled_log_net_worth_per_capita_bar.png')

#================================================================
# Scaled Per-capita log wealth by percentile category (bar graphs + inset)
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

# Plot the bars
plt.figure(figsize=(14, 8))

palette_colors = sns.color_palette()  # Get the palette colors
last_color = palette_colors[-1]  # Get the last color from the palette
for i, category in enumerate(fed_data.POPULATION_SIZES.keys()):
	if i == len(fed_data.POPULATION_SIZES.keys()) - 1:  # Check if it's the last bar
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge', edgecolor=last_color, linewidth=1.5)
	else:
		plt.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge')

# Add a title and labels
plt.title('Scaled Per-capita log Wealth Distribution by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])

modified_percentiles_str_list = fed_data.PERCENTILES_STR_LIST[:-1]
modified_percentiles_str_list[-1] = '99-99.99\n99.99-100'

# Set the x-ticks to be in the middle of each bar for clarity. Remove the very last label
plt.xticks(ticks=[left + (width/2) for left, width in zip(bars_left, bars_width)][:-1], 
           labels=modified_percentiles_str_list)
plt.yscale('log')  

plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

ax = plt.gca()

# Define position and size of the inset plot: [x, y, width, height]
ax_inset = ax.inset_axes([0.5, 0.6, 0.25, 0.35], xlim=[99.0, 100.1])  
ax_inset.set_yscale('log')

for i, category in enumerate(fed_data.POPULATION_SIZES.keys()):
    ax_inset.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], align='edge')

# Highlight the zoomed area in the main plot using a rectangle
# rect = patches.Rectangle((90, 0), 10, max(normalized_wealth.values), edgecolor='red', facecolor='none')
# plt.gca().add_patch(rect)

rect, connecting_lines = ax.indicate_inset_zoom(ax_inset, edgecolor="red")
lower_left_line, upper_left_line, lower_right_line, upper_right_line = connecting_lines

save_fig(plt, 'scaled_log_net_worth_per_capita_bar+inset.png')


#================================================================
# Scaled Per-capita wealth by granular percentile (bar graphs)
#================================================================

# Extract values into new df
values_df = pd.DataFrame({'values': normalized_wealth.values})

# Calculate the percentiles
# weibull for heavy-tailed distribution
normalized_wealth_percentiles = calculate_percentiles(values_df, 'values', 1, interpolation='linear')

# Increase number of colors
sns.set_palette(sns.dark_palette("#69d", n_colors=101, reverse=False))

# Plot the bars
plt.figure(figsize=(14, 8))
for percentile, net_wealth in normalized_wealth_percentiles.items():
	plt.bar(x=percentile, height=net_wealth, width=1, label=percentile, align='edge')

# Add a title and labels
plt.title('Scaled Per-capita Wealth Distribution by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])

plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'scaled_net_worth_per_capita_bar_granular.png')


#================================================================
# Scaled Per-capita log wealth by granular percentile (bar graphs)
#================================================================

# Extract values into new df
values_df = pd.DataFrame({'values': normalized_wealth.values})

# Calculate the percentiles
# weibull for heavy-tailed distribution
normalized_wealth_percentiles = calculate_percentiles(values_df, 'values', 1, interpolation='linear')

# Increase number of colors
sns.set_palette(sns.dark_palette("#69d", n_colors=101, reverse=False))

# Plot the bars
plt.figure(figsize=(14, 8))
for percentile, net_wealth in normalized_wealth_percentiles.items():
	plt.bar(x=percentile, height=net_wealth, width=1, label=percentile, align='edge')

# Add a title and labels
plt.title('Scaled Per-capita log Wealth Distribution by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])

plt.yscale('log')  

plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'scaled_log_net_worth_per_capita_bar_granular.png')


#================================================================
# Scaled Per-capita log-log wealth by granular percentile (bar graphs)
#================================================================

# Extract values into new df
values_df = pd.DataFrame({'values': normalized_wealth.values})

# Calculate the percentiles
# weibull for heavy-tailed distribution
normalized_wealth_percentiles = calculate_percentiles(values_df, 'values', 1, interpolation='linear')

# Increase number of colors
sns.set_palette(sns.dark_palette("#69d", n_colors=101, reverse=False))

# Plot the bars
plt.figure(figsize=(14, 8))
for percentile, net_wealth in normalized_wealth_percentiles.items():
	plt.bar(x=percentile, height=net_wealth, width=1, label=percentile, align='edge')

# Add a title and labels
plt.title('Scaled Per-capita log-log Wealth Distribution by Population Percentile')
plt.ylabel('Net Worth per Capita (Millions)')
plt.xlabel('Population Percentile')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()
# plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])

plt.yscale('log')  
plt.xscale('log')

# Format the ticks to display as standard numbers
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())

# Ensure all ticks are shown (not just the major ones)
plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())



plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_fig(plt, 'scaled_log_log_net_worth_per_capita_bar_granular.png')

exit()

#-----------------------------------------------
# Scaled-log normalized wealth line graph
#-----------------------------------------------
plt.figure(figsize=(14, 8))

x_space = np.linspace(0, 100, 500)

# print(normalized_wealth.values[::-1])
y_interp = np.interp(x_space, [size*100 for _, size in fed_data.population_sizes.items()], normalized_wealth.values[::-1])
plt.plot(x_space, y_interp, color='red')
plt.title('log-log Scaled Normalized Wealth Distribution in 2020Q1 by Population Percentile')
plt.ylabel('Net Worth per Person (Millions)')
plt.xlabel('Population Percentile')
plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xscale('log')
ax = plt.gca()
# ax.set_ylim(0, 35)
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_log-log-scaled_wealth_normalized_line.png')
plt_cnt += 1

#-----------------------------------------------
# Polynomial interpolated Scaled per-capita wealth line graph
#-----------------------------------------------
x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [size*100 for _, size in fed_data.population_sizes.items()], normalized_wealth.values[::-1])

# Polynomial interpolation
degrees = np.arange(1, 10)  # This sets the max degree to 9, you can change this value.
best_degree = 0
min_residual = float('inf')

for deg in degrees:
    p = np.polyfit(x_space, y_interp, deg)
    y_poly = np.polyval(p, x_space)
    residual = np.sum((y_interp - y_poly)**2)
    if residual < min_residual:
        min_residual = residual
        best_degree = deg

p_best = np.polyfit(x_space, y_interp, best_degree)
y_best_poly = np.polyval(p_best, x_space)

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red', label='Data')
plt.plot(x_space, y_best_poly, color='blue', linestyle='--', label=f'Polynomial (Degree {best_degree})')

plt.title('Scaled Per-capita Wealth Distribution in 2020Q1 by Population Percentile')
plt.ylabel('Net Worth per Person (Millions)')
locs, labels = plt.yticks()
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])
plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.savefig(f'out/{plt_cnt}_per_capita_wealth_polynomial_fit_line.png')
plt_cnt += 1


# #-----------------------------------------------
# # Scaled wealth line graph
# #-----------------------------------------------
# plt.figure(figsize=(14, 8))

# x_space = np.linspace(0, 100, 500)
# print(data_2020Q1.values[::-1])
# y_interp = np.interp(x_space, [size*100 for _, size in fed_data.population_sizes.items()], data_2020Q1.values[::-1])
# plt.plot(x_space, y_interp, color='red')
# plt.title('log-log Scaled Normalized Wealth Distribution in 2020Q1 by Population Percentile')
# plt.ylabel('Net Worth per Person (Millions)')
# plt.xlabel('Population Percentile')
# # plt.yscale('log')  # This sets the y-axis to a logarithmic scale
# # plt.xscale('log')
# ax = plt.gca()
# # ax.set_ylim(0, 35)
# plt.xticks(rotation=45)
# # plt.gca().invert_xaxis()
# # plt.legend(loc="upper left", bbox_to_anchor=(1,1))
# plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.savefig(f'out/{plt_cnt}_scaled_wealth_line.png')
# plt_cnt += 1



