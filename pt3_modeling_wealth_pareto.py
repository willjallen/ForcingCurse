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

#================================================================
# Output Directory Setup
#================================================================

OUTPUT_DIRECTORY = 'out/pt3'

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

# Use Seaborn's blue color palette
sns.set_palette(sns.dark_palette("#69d", reverse=False))  

 
 
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

psid_chosen_period_df = psid_wealth_dict[PSID_CHOSEN_PERIOD]
psid_wealth_chosen_period_df = psid_chosen_period_df['IMP WEALTH W/ EQUITY']

#================================================================
# Calculate percentiles
#================================================================

wealth_percentiles = calculate_percentiles(psid_chosen_period_df, 'IMP WEALTH W/ EQUITY', 0.01)
percentiles_df = pd.DataFrame(list(wealth_percentiles.items()), columns=['Percentile', 'Wealth'])

#================================================================
# Pareto CDF
#================================================================
#region
# Plotting the data

space = np.linspace(1, 5, 1000)
pareto_cdf_a_inf = pareto.cdf(space, float('inf'), scale=1)
pareto_cdf_a_3 = pareto.cdf(space, 3, scale=1)
pareto_cdf_a_2 = pareto.cdf(space, 2, scale=1)
pareto_cdf_a_1_16 = pareto.cdf(space, 1.16, scale=1)
pareto_cdf_a_1 = pareto.cdf(space, 1, scale=1)



# Set up figure
plt.figure(figsize=(14,8))

# Plot
plt.plot(space, pareto_cdf_a_inf, label=r'$\alpha=\infty$')
plt.plot(space, pareto_cdf_a_3, label=r'$\alpha=3$')
plt.plot(space, pareto_cdf_a_2, label=r'$\alpha=2$')
plt.plot(space, pareto_cdf_a_1_16, label=r'$\alpha=1.16$')
plt.plot(space, pareto_cdf_a_1, label=r'$\alpha=1$')

# Title and labels
plt.title('Pareto Distribution CDF')
plt.ylabel(r'Pr$(X \leq x)$')
plt.xlabel(r'$x$')

# Plot properties
plt.legend(prop={'size': 12}) 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'pareto_cdf.png')

#endregion

#================================================================
# Pareto PDF
#================================================================
#region
# Plotting the data

space = np.linspace(1, 5, 1000)
# pareto_pdf_a_inf = pareto.pdf(space, float('inf'), scale=1)
pareto_pdf_a_3 = pareto.pdf(space, 3, scale=1)
pareto_pdf_a_2 = pareto.pdf(space, 2, scale=1)
pareto_pdf_a_1_16 = pareto.pdf(space, 1.16, scale=1)
pareto_pdf_a_1 = pareto.pdf(space, 1, scale=1)

# Set up figure
plt.figure(figsize=(14,8))

# Plot
plt.vlines(1, 0, 3, colors='black', linestyles='solid')

# Plot the intersection points as circles
palette = sns.color_palette()
colors = iter(palette)

# plt.plot(space, pareto_pdf_a_inf, label=r'$\alpha=\infty$')
plt.plot([], [], label=r'$\lim_{\alpha \to \infty} = \delta(x - x_m)$', color='none')
plt.plot(space, pareto_pdf_a_3, label=r'$\alpha=3$')
plt.plot(space, pareto_pdf_a_2, label=r'$\alpha=2$')
plt.plot(space, pareto_pdf_a_1_16, label=r'$\alpha=1.16$')
plt.plot(space, pareto_pdf_a_1, label=r'$\alpha=1$')

# Calculate y-values at x = 1
y_values = [pareto.pdf(1, a, scale=1) for a in [3, 2, 1.16, 1]]

# Reset the color iterator
palette = sns.color_palette()
colors = iter(palette)

# Plot the intersection points with corresponding colors
for y in y_values:
    plt.scatter(1, y, color=next(colors), s=50, zorder=10)

# Title and labels
plt.title('Pareto Distribution PDF')
plt.ylabel(r'Pr$(X = x)$')
plt.xlabel(r'$x$')

# Plot properties
plt.legend(prop={'size': 12}) 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt)

# Save
save_fig(plt, 'pareto_pdf.png')

#endregion

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
# Pareto/Distribution fitting
#================================================================

# First we will ignore all values <= 0

'''
Nature of the Data: Wealth distribution is often multimodal and not well-captured by simple distributions. 
A Pareto distribution assumes that the log-log plot of the cumulative distribution is linear, which may not be 
the case for the entire range of data, particularly near the lower end.
'''

#================================================================
# Clamped >0 log-log Net Worth emperical CDF 
#================================================================
#region
arr = psid_wealth_chosen_period_df 
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

#================================================================
# log-log Net Worth emperical CDF, clamped range [1, 1_000_000_000], linear regressions
#================================================================
#region

# Function to perform linear regression and return the fit statistics
def perform_linear_regression(x, y):
	slope, intercept, r_value, _, _ = linregress(x, y)
	return slope, intercept, r_value**2

# Function to calculate the weighted R^2
def weighted_r_squared(r_squared, num_points, total_points):
	return r_squared * (num_points / total_points)**(1/5)

arr = psid_wealth_chosen_period_df 
m = 1
n = 100_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'{inclusion_ratio}%')

# Filter the values we want
filtered_arr = arr[(arr > m) & (arr <= n)]

# Sort the data for the empirical CDF
sorted_data = np.sort(filtered_arr)

# Calculate the empirical CDF values
cdf_values = np.arange(1, len(sorted_data)+1) / len(sorted_data)


# Calculate the log of the minimum and maximum net worth values for the fitting range
log_min_value = np.log(m)
log_max_value = np.log(n)

# Generate a set of evenly spaced points in log space for the start of the fits
log_space = np.linspace(log_min_value, log_max_value, 19)

# Exponentiate these points to obtain the actual net worth values for the start of the fits
fit_starts = np.exp(log_space)

# Initialize the plot
plt.figure(figsize=(14, 8))

# Plot the empirical CDF
plt.plot(sorted_data, cdf_values, marker='.', linestyle='none', markersize=5, label='Empirical CDF')

# Fit lines and goodness of fit measures
fit_lines = []
goodness_of_fit = []
weighted_goodness_of_fit = []

# Fit space
fit_space = np.linspace(0.05, 0.95, 19)

# Perform linear regression from different points in log space
for start_value in fit_starts:
	# Find the index where the sorted data exceeds the start value
	start_index = np.searchsorted(sorted_data, start_value)
	if start_index < len(sorted_data) - 1:  # Ensure we have at least two points to fit
		# Perform linear regression on log-log scale from this index to the end
		slope, intercept, r_squared = perform_linear_regression(
			np.log(sorted_data[start_index:]), 
			np.log(cdf_values[start_index:])
		)
		goodness_of_fit.append(r_squared)
		fit_lines.append((slope, intercept))
		num_points_in_fit = len(sorted_data) - start_index
		weighted_r2 = weighted_r_squared(r_squared, num_points_in_fit, len(sorted_data))
		weighted_goodness_of_fit.append(weighted_r2)

# Sort the lines by weighted r^2
sorted_lines_with_weighted_r2 = sorted(
	zip(fit_lines, goodness_of_fit, weighted_goodness_of_fit, fit_starts), 
	key=lambda x: x[2], 
	reverse=True
)


# Set up figure
plt.figure(figsize=(14, 8))

# Plot
# Emperical CDF
plt.plot(sorted_data, cdf_values, marker='.', linestyle='none', markersize=5, label='Empirical CDF')

# Linear fits

# Get the index of the best fit
best_fit_index = np.argmax(weighted_goodness_of_fit)

# Color scheme for red gradient
red_colors = plt.cm.Reds(np.linspace(0.3, 1, len(sorted_lines_with_weighted_r2)))

# Plot the fit lines
for i, ((slope, intercept), r2, weighted_r2, start_value) in enumerate(sorted_lines_with_weighted_r2):
	# Calculate the range for the linear fit
	start_index = np.searchsorted(sorted_data, start_value)
	end_index = len(sorted_data) - 1
	fit_range = sorted_data[start_index:end_index+1]
	
	# Generate the y-values for the fit line within the defined range
	fit_line = np.exp(intercept + slope * np.log(fit_range))
	
	# Determine the color of the line
	color = 'purple' if i == 0 else red_colors[i]  # First line is the best fit
	
	# Plot the line with the appropriate label and alpha based on the weighted goodness of fit
	plt.plot(fit_range, fit_line, label=f'Fit {i+1} ({fit_range[0]:.0f}, {fit_range[-1]:.0f}) R^2={r2:.3f}',
			 color=color, alpha=r2)

	
# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD} - Empirical CDF of {"Household" if HOUSEHOLD else "Individual"} Net Worth with Linear Fits')
plt.ylabel('CDF (Proportion less than x)')
plt.xlabel('Net Worth')

# y-axis
plt.yscale('log')

# x-axis
plt.xticks(rotation=45)
plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "${:,.0f}".format(x)))


# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Notate
notate_plot(plt, note="data clamped to range [1, 100,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_log_log_cdf_lin_fit_plot.png')

#endregion

#================================================================
# Pareto CDF, clamped range [1, 1_00_000_000]
#================================================================
#region

#-------------------------------------
# Extract Data
#-------------------------------------
arr = psid_wealth_chosen_period_df 
m = 1
n = 100_000_000
# inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
# print(f'{inclusion_ratio}%')

filtered_arr = arr[(arr >= m) & (arr <= n)]
sorted_data = np.sort(filtered_arr)

#-------------------------------------
# Pareto CDF
#-------------------------------------

# Estimate parameters for the Pareto distribution
shape, location, scale = pareto.fit(sorted_data, 0.7, loc=1, scale=1)
print("here")
print(shape, location, scale)

# Generate cdf
pareto_cdf = pareto.cdf(sorted_data, shape, loc=location, scale=scale)

#-------------------------------------
# Emperical CDF
#-------------------------------------

# Calculate the empirical CDF values
emperical_cdf_values = np.arange(1, len(sorted_data)+1) / len(sorted_data)


# Set up figure
plt.figure(figsize=(14, 8))

# Plot
plt.plot(sorted_data, emperical_cdf_values, marker='.', linestyle='none', markersize=5, label='Empirical CDF')

# Plot
plt.plot(sorted_data, pareto_cdf, label='Pareto CDF', color='orange')

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD}' + f' - Misleading {"Household" if HOUSEHOLD else "Individual"} Net Worth Pareto CDF fit,'+ r' $\alpha =$' +f'{shape:,.2f},'+ f' location = {location:,.2f}' + r', $x_m =$' + f'{scale:,.2f}')
plt.ylabel(r'Pr$(X \leq x)$')
plt.xlabel('Net Worth')

# y-axis
# plt.yscale('log')

# x-axis
plt.xscale('log')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "${:,.0f}".format(x)))

# Plot properties
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, note="data clamped to range [1, 100,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_plot_pareto_cdf.png')

#endregion

#================================================================
# Pareto PDF, Net Worth histogram, 200 bins, clamped range (1, 1_000_000_000]
#================================================================
#region
arr = psid_wealth_chosen_period_df 
m = 1
n = 100_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'{inclusion_ratio}%')

# Calculate the total range for the bins
total_range = n - m

# Number of bins
num_bins = 800

x_grid = np.linspace(0, n, 1000)
filtered_arr = arr[(arr > m) & (arr <= n)]

# Filter the data to get the tail (upper 20% by default)
# tail_data = np.sort(filtered_arr)[int(0.8 * len(filtered_arr)):]

# Calculate bin width
bin_width = total_range / num_bins

# The upper limit of the first bin (rightmost corner)
first_bin_upper_limit = m + bin_width
print(first_bin_upper_limit)

# Filter the data to get the tail (>= first bin)
tail_data = np.sort(filtered_arr[filtered_arr >= 1])


# Estimate parameters for the Pareto distribution using MLE
shape, location, scale = pareto.fit(tail_data)
print(shape, location, scale)
# Generate the PDF of the fitted Pareto distribution for plotting
pareto_pdf_mle = pareto.pdf(x_grid, shape, loc=location, scale=scale)

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
# Looks way better with a log plot
count, bins, _ = plt.hist(filtered_arr, bins=num_bins, range=(m, n), log=True, histtype='stepfilled',linewidth=0 , alpha=0.8)
# print(count, bins)
plt.plot(x_grid, pareto_pdf_mle * len(tail_data) * np.diff(bins)[0], label='Pareto fit (MLE)', color='orange')

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD}' + f' - Misleading {"Household" if HOUSEHOLD else "Individual"} Net Worth Pareto PDF fit,'+ r' $\alpha =$' +f'{shape:,.2f},'+ f' location = {location:,.2f}' + r', $x_m =$' + f'{scale:,.2f}')
plt.ylabel('Frequency')
plt.xlabel('Net Worth')

# y-axis

# x-axis
plt.xticks(rotation=45)
locs, labels = plt.xticks()  # Get current y-axis tick locations and labels
plt.xticks(locs, [f"${x*1e-6:.1f}M" for x in locs])  # Set new labels in millions
plt.xlim(m, n)

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, note="data clamped to range [1, 100,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_hist_pareto_pdf.png')

#endregion

#================================================================
# Pareto PPF, clamped range [1, 1_00_000_000]
#================================================================
#region
arr = psid_wealth_chosen_period_df 
m = 1
n = 100_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'{inclusion_ratio}%')

# Calculate the total range for the bins
total_range = n - m

# Number of bins
num_bins = 800

x_grid = np.linspace(0, n, 1000)
filtered_arr = arr[(arr > m) & (arr <= n)]

# Filter the data to get the tail (upper 20% by default)
# tail_data = np.sort(filtered_arr)[int(0.8 * len(filtered_arr)):]

# Calculate bin width
bin_width = total_range / num_bins

# The upper limit of the first bin (rightmost corner)
first_bin_upper_limit = m + bin_width
print(first_bin_upper_limit)

# Filter the data to get the tail (>= first bin)
tail_data = np.sort(filtered_arr[filtered_arr >= 1])


# Estimate parameters for the Pareto distribution using MLE
shape, location, scale = pareto.fit(tail_data)
print(shape, location, scale)

percentiles = np.linspace(0.0, 1.0, 1000)
pareto_ppf = pareto.ppf(percentiles, shape, loc=location, scale=scale)

# Set up figure
plt.figure(figsize=(14, 8))

# Plot
plt.plot(percentiles*100, pareto_ppf, color='orange')

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD}' + f' - Misleading {"Household" if HOUSEHOLD else "Individual"} Net Worth Pareto Percentiles,'+ r' $\alpha =$' +f'{shape:,.2f},'+ f' location = {location:,.2f}' + r', $x_m =$' + f'{scale:,.2f}')
plt.ylabel('Net Worth')
plt.xlabel('Percentile')

# y-axis
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "${:,.0f}".format(x)))

# x-axis
# plt.xticks(rotation=45)
plt.xticks(np.arange(0, 101, 10))

# Plot properties
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, note="data clamped to range [1, 100,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_plot_pareto_ppf.png')

#endregion

#================================================================
# Comparing with FED percentile data
#================================================================
#region
fed_data = FedData()
fed_data.load()
net_worth_df = fed_data.get_net_worth_data()

chosen_period = '2019Q1'
net_worth_chosen_period_df = net_worth_df.loc[chosen_period]

# Define the total population
TOTAL_POPULATION = 333_287_557

# Calculate the number of people in each category
people_in_category = {category: TOTAL_POPULATION * size for category, size in fed_data.POPULATION_SIZES.items()}

# Normalize the wealth by number of people in each category
normalized_wealth = net_worth_chosen_period_df / pd.Series(people_in_category)

#endregion

#================================================================
# Pareto PPF, clamped range (0, 1_000_000_000]
#================================================================
#region
arr = psid_wealth_chosen_period_df 
m = 1
n = 100_000_000
inclusion_ratio = np.sum((arr > m) & (arr < n)) / len(arr)
print(f'{inclusion_ratio}%')

# Calculate the total range for the bins
total_range = n - m

# Number of bins
num_bins = 800

x_grid = np.linspace(0, n, 1000)
filtered_arr = arr[(arr > m) & (arr <= n)]

# Filter the data to get the tail (upper 20% by default)
# tail_data = np.sort(filtered_arr)[int(0.8 * len(filtered_arr)):]

# Calculate bin width
bin_width = total_range / num_bins

# The upper limit of the first bin (rightmost corner)
first_bin_upper_limit = m + bin_width
print(first_bin_upper_limit)

# Filter the data to get the tail (>= first bin)
tail_data = np.sort(filtered_arr[filtered_arr >= 1])


# Estimate parameters for the Pareto distribution using MLE
shape, location, scale = pareto.fit(tail_data)
print(shape, location, scale)

percentiles = np.linspace(0.0, 1.0, 1000)
pareto_ppf = pareto.ppf(percentiles, shape, loc=location, scale=scale)

# Set up figure
plt.figure(figsize=(14, 8))

print(normalized_wealth)

# Generate the colors for each set of plots
# left_colors = sns.color_palette("flare", n_colors=5)
# mid_colors = sns.color_palette("crest", n_colors=5)
colors = sns.color_palette("viridis", n_colors=5)

# Now loop through and plot each category with its respective colors
for i, (category, (start, end)) in enumerate(fed_data.PERCENTILES.items()):
	plt.plot(start, normalized_wealth[category], marker='o', label=f'{fed_data.PERCENTILES_STR[category]} Left', color=colors[i])

for i, (category, (start, end)) in enumerate(fed_data.PERCENTILES.items()):
	plt.plot(start + (end-start)/2, normalized_wealth[category], marker='o', label=f'{fed_data.PERCENTILES_STR[category]} Mid', color=colors[i])
 
for i, (category, (start, end)) in enumerate(fed_data.PERCENTILES.items()):
	plt.plot(end, normalized_wealth[category], marker='o', label=f'{fed_data.PERCENTILES_STR[category]} Right', color=colors[i])

# Plot
plt.plot(percentiles*100, pareto_ppf, color='orange')

# Title and labels
plt.title(f'{PSID_CHOSEN_PERIOD}' + f' - Misleading {"Household" if HOUSEHOLD else "Individual"} Net Worth Pareto Percentiles; Fed comparison,'+ r' $\alpha =$' +f'{shape:,.2f},'+ f' location = {location:,.2f}' + r', $x_m =$' + f'{scale:,.2f}')
plt.ylabel('Net Worth')
plt.xlabel('Percentile')

# y-axis
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "${:,.0f}".format(x)))

# x-axis
# plt.xticks(rotation=45)
plt.xticks(np.arange(0, 101, 10))

# Plot properties
handles, labels = plt.gca().get_legend_handles_labels()

# Define a custom legend handler by creating a dummy line object
divider_line = mlines.Line2D([], [], color='black', linestyle='--')

# Function to insert a divider
def insert_divider(index):
    handles.insert(index, divider_line)
    labels.insert(index, '')  # An empty string for the label

insert_divider(5)  
insert_divider(11)

plt.legend(handles, labels)

plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Notate
notate_plot(plt, data_source="simba.isr.umich.edu\nfederalreserve.gov", note="data clamped to range [1, 100,000,000]")

# Save
save_fig(plt, 'net_worth_clamped_plot_pareto_ppf_fed_comparison.png')

#endregion