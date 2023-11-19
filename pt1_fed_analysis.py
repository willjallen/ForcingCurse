import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import seaborn as sns
import os
import shutil
from io import StringIO
from data import FedData, PSIDData
from utils.helper import calculate_percentiles
import math

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
# Plot notator wrapper function
#================================================================

def notate_plot(plt: plt, data_source="federalreserve.gov", website="wallen.me/projects/modeling-wealth", note="", margin=0.18):
	# Adjust the bottom margin to make space for the note
	plt.subplots_adjust(bottom=margin)

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

# Use Seaborn's blue color palette
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
plt.title('Distribution of Household Net Worth 1989-2023')
plt.ylabel('Net Worth (Trillions)')
plt.xlabel('Date')

# y-axis
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'net_worth_over_time_stacked.png')

#================================================================
# Picking out a single period
#================================================================

chosen_period = '2019Q1'
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
plt.title(f'{chosen_period} - Household Net Worth by Population Percentile')
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

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'net_worth_dist_bar.png')

#================================================================
# Calculate normalized (Per capita) wealth
#================================================================

# Define the total population
TOTAL_POPULATION = 333_287_557

# Calculate the number of people in each category
people_in_category = {category: TOTAL_POPULATION * size for 
					  category, size in fed_data.POPULATION_SIZES.items()}

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
plt.title(f'{chosen_period} - Household Net Worth Per capita by Population Percentile')
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

# Notate
notate_plot(plt)

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
# plt.subplots_adjust(bottom=0.2)
# Add note about exaggeration
plt.figtext(.68, .8, "linewidth of 99.99-100 percentile is exaggerated by 5x")
# plt.text(0.5, 0.02, 'Note: the linewidth of 99.99-100 percentile is exaggerated by 5x compared to the original width', 
		#  ha='center', va='center', transform=plt.gcf().transFigure)


# Title and labels
plt.title(f'{chosen_period} - Household Net Worth Per capita by Population Percentile with Proportional Scaling')
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

# Notate
notate_plot(plt)

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
plt.title(f'{chosen_period} - Household Net Worth Per capita by Population Percentile')
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

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'log_net_worth_per_capita_bar.png')

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
# plt.subplots_adjust(bottom=0.2)
# # Add note about exaggeration
plt.figtext(.68, .8, "linewidth of 99.99-100 percentile is exaggerated by 5x")
# plt.text(0.5, 0.02, 'Note: the linewidth of 99.99-100 percentile is exaggerated by 5x compared to the original width', 
# 		 ha='center', va='center', transform=plt.gcf().transFigure)

# Title and labels
plt.title(f'{chosen_period} - Household Net Worth Per capita by Population Percentile with Proportional Scaling')
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

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'scaled_log_net_worth_per_capita_bar.png')

#================================================================
# Animation: Scaled Per capita log wealth by percentile category (bar graphs) 
#================================================================
people_in_category = {category: TOTAL_POPULATION * size for category, size in fed_data.POPULATION_SIZES.items()}

def update(frame, ax1, ax2, ax3, ax4):
	ax1.clear()
	ax2.clear() 
	ax3.clear()
	ax4.clear()

	year, quarter = frame
	chosen_period = f'{year}{quarter}'
	net_worth_chosen_period_df = net_worth_df.loc[chosen_period]

	# Calculate the normalized (Per capita) wealth
	normalized_wealth = net_worth_chosen_period_df / pd.Series(people_in_category)

	#---------------------------------
	# Net Worth Plot
	#---------------------------------
	for category, (start, end) in fed_data.PERCENTILES.items():
		ax1.bar(f"{start}-{end}", net_worth_chosen_period_df[category], label=category)

	ax1.set_title(f'{chosen_period} - Household Net Worth by Population Percentile')
	ax1.set_ylabel('Net Worth (Trillions)')
	ax1.set_xlabel('Population Percentile')

	# y-axis
	locs = ax1.get_yticks()  # Get current y-axis tick locations and labels
	ax1.set_yticks(locs, [f"${x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions


	# x-axis
	ax1.set_xticks(range(len(fed_data.PERCENTILES)))
	ax1.set_xticklabels([f"{start}-{end}" for start, end in fed_data.PERCENTILES.values()], rotation=45)

	# plot properties
	ax1.grid(True, axis='y', linestyle='--', linewidth=0.5)

	#---------------------------------
	# Net Worth Per Capita Plot
	#---------------------------------
	for category, (start, end) in fed_data.PERCENTILES.items():
		ax2.bar(f"{start}-{end}", normalized_wealth[category], label=category)

	ax2.set_title(f'{chosen_period} - Household Net Worth Per Capita by Population Percentile')
	ax2.set_ylabel('Net Worth per Capita (Millions)')
	ax2.set_xlabel('Population Percentile')
	
	# y-axis
	locs = ax2.get_yticks()  # Get current y-axis tick locations and labels
	ax2.set_yticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

	# x-axis
	ax2.set_xticks(range(len(fed_data.PERCENTILES)))
	ax2.set_xticklabels([f"{start}-{end}" for start, end in fed_data.PERCENTILES.values()], rotation=45)

	# plot properties
	ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)
	
	#---------------------------------
	# Log Net Worth Per Capita Plot
	#--------------------------------- 
	# Plot
	for category, (start, end) in fed_data.PERCENTILES.items():
		ax3.bar(f"{start}-{end}", normalized_wealth[category], label=category)
	# plt.plot(fed_data.PERCENTILES_STR_LIST, normalized_wealth.values, color='red')

	# Title and labels
	ax3.set_title(f'{chosen_period} - Household Net Worth Per capita by Population Percentile')
	ax3.set_ylabel('Net Worth per Capita')
	ax3.set_xlabel('Population Percentile')

	# y-axis
	ax3.set_yscale('log')  

	# Currency formatter function
	def currency_formatter(x, pos):
		return "${:,.0f}".format(x)

	# Set the y-axis formatter
	ax3.yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

	# x-axis
	ax3.set_xticks(range(len(fed_data.PERCENTILES)))
	ax3.set_xticklabels([f"{start}-{end}" for start, end in fed_data.PERCENTILES.values()], rotation=45)

	# Plot properties
	ax3.grid(True, axis='y', linestyle='--', linewidth=0.5)

	#---------------------------------
	# Scaled Log Net Worth Per Capita Plot
	#--------------------------------- 
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

	# Plot
	palette_colors = sns.color_palette()  # Get the palette colors
	last_color = palette_colors[-1]  # Get the last color from the palette
	for i, category in enumerate(fed_data.POPULATION_SIZES.keys()):
		if i == len(fed_data.POPULATION_SIZES.keys()) - 1:  # Check if it's the last bar
			ax4.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge', edgecolor=last_color, linewidth=0.5)
		else:
			ax4.bar(x=bars_left[i], height=normalized_wealth[category], width=bars_width[i], label=category, align='edge')

	# Adjust the bottom margin to make space for the note
	# ax4.subplots_adjust(bottom=0.2)
	# # Add note about exaggeration
	# ax4.text(.68, .8, "linewidth of 99.99-100 percentile is exaggerated by 5x")
	# ax4.text(0.5, 0.02, 'Note: the linewidth of 99.99-100 percentile is exaggerated by 5x compared to the original width', 
	# 		 ha='center', va='center', transform=ax4.gcf().transFigure)

	# Title and labels
	ax4.set_title(f'{chosen_period} - Household Net Worth Per capita by Pop. Percentile w/ Prop. Scaling')
	ax4.set_ylabel('Net Worth per Capita')
	ax4.set_xlabel('Population Percentile')

	# y-axis
	ax4.set_yscale('log')  

	def currency_formatter(x, pos):
		return "${:,.0f}".format(x)

	ax4.yaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))

	# x-axis
	modified_percentiles_str_list = fed_data.PERCENTILES_STR_LIST[:-1]
	modified_percentiles_str_list[-1] = '99-99.99\n99.99-100'

	# Set the x-ticks to be in the middle of each bar for clarity. Remove the very last label
	ax4.set_xticks(ticks=[left + (width/2) for left, width in zip(bars_left, bars_width)][:-1], 
			labels=modified_percentiles_str_list, rotation=45)
	# ax4.set_xticks(rotation=45)

	# Plot properties
	ax4.grid(True, axis='y', linestyle='--', linewidth=0.5)
 
	plt.tight_layout() 
	notate_plot(plt, margin=0.20)


fig = plt.figure(figsize=(14, 8))

# Create a list of all year-quarter combinations
years = range(1989, 2024)
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
frames = [(year, quarter) for year in years for quarter in quarters 
		  if not (year == 1989 and quarter in ['Q1', 'Q2']) 
		  and not (year == 2023 and quarter in ['Q3', 'Q4'])]

# Set up subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))

print('Rendering animation...')
anim = animation.FuncAnimation(fig, update, fargs=(ax1, ax2, ax3, ax4), frames=frames, interval=100)
anim.save('out/pt1/net_worth_animation.mp4', writer='ffmpeg')
print('Animation finished rendering.')
