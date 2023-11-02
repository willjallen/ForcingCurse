import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from io import StringIO

# https://www.federalreserve.gov/releases/z1/dataviz/dfa/distribute/chart/#range:1989.3,2023.2;quarter:135;series:Net%20worth;demographic:networth;population:all;units:levels
# The fed derives quarterly results by interpolating the consumer report data done every 3ish years

# https://dqydj.com/net-worth-percentiles/
# https://dqydj.com/millionaires-in-america/
# https://www.kaggle.com/datasets/prasertk/forbes-worlds-billionaires-list-2022

# Household wealth data
# https://simba.isr.umich.edu/DC/s.aspx
# WEALTH W/ 
# found from https://arxiv.org/pdf/1209.4787.pdf
# section 6.1


# Pareto:
# Simple Models of Pareto Income and Wealth Inequality
# https://assets.aeaweb.org/asset-server/articles-attachments/jep/app/2901/29010029_app.pdf 
# Pareto Models for Top Incomes
# https://hal.science/hal-02145024v1/file/TopIncomes.pdf
# The Kinetics of Wealth and the Origin of the Pareto Law
# https://arxiv.org/pdf/1212.6300.pdf
if os.path.exists('/out'):
	shutil.rmtree('out')
 
os.makedirs('out')
 
#=============================================
# Importing Data
#=============================================

# Set the Seaborn style
sns.set_style("darkgrid")

# Set global defaults for matplotlib
plt.rcParams['savefig.dpi'] = 300  # set the DPI for saved figures

# Use Seaborn's green color palette
sns.set_palette(sns.dark_palette("#69d")) 


plt_cnt = 0

# # Normalize the net worth data by population size for each category
# for category, size in population_sizes.items():
#     df_net_worth[category] *= size

# Calculate total wealth for each time period (sum across all categories)
# df_net_worth['Total Net Worth'] = df_net_worth.sum(axis=1)

# Inspect the transformed dataframe
# print(df_net_worth.head())

#=============================================
# Graph of cumulative wealth over time over net worth percentiles
#=============================================

# Plotting the data

# Make a copy to change the labels
df_net_worth_copy = df_net_worth.copy()

df_net_worth_copy.columns = percentiles_str.values()

plt.figure(figsize=(14,8))
df_net_worth_copy.plot.area(ax=plt.gca())
ax=plt.gca()
ax.legend(title="Population Percentiles")
plt.title('Distribution of Net Worth 1989-2023')
plt.ylabel('Net Worth (Trillions)')

# Adjust the y-axis to be in trillions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

plt.xlabel('Date')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_dist_net_worth_over_time.png')
plt_cnt += 1

# Set color pallete back to reverse
sns.set_palette(sns.dark_palette("#69d", reverse=True)) 


#=============================================
# Picking out a single year
#=============================================

data_2020Q1 = df_net_worth.loc['2020Q1']
print(data_2020Q1)

# Extracting the data for 2020Q1 excluding 'Total Net Worth'
# data_2020Q1 = data_2020Q1.drop('Total Net Worth')

#-----------------------------------------------
# Plotting wealth based on percentiles (bar graphs)
#-----------------------------------------------

# Plotting the data
plt.figure(figsize=(14, 8))
for category, (start, end) in percentiles.items():
    plt.bar(f"{start}-{end}", data_2020Q1[category], label=category)

plt.title('Net Worth Distribution in 2020Q1 by Population Percentile')

# Adjust the y-axis to be in trillions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-12:.1f}T" for x in locs])  # Set new labels in trillions

plt.ylabel('Net Worth (Trillions)')
plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_net_worth_dist_bar.png')
plt_cnt += 1

#-----------------------------------------------
# Plotting wealth based on normalized percentiles (bar + line graphs)
#-----------------------------------------------

# Define the total population
total_population = 333_287_557

# Calculate the number of people in each category
people_in_category = {category: total_population * size for category, size in population_sizes.items()}

# Normalize the wealth by number of people in each category
normalized_wealth = data_2020Q1 / pd.Series(people_in_category)

print(normalized_wealth.head())

# Plotting the bar and line graph
plt.figure(figsize=(14, 8))
for category, (start, end) in percentiles.items():
    plt.bar(f"{start}-{end}", normalized_wealth[category], label=category)
# normalized_wealth.plot(color='red')
plt.title('Per-capita Wealth Distribution in 2020Q1 by Population Percentile')
plt.ylabel('Net Wealth per Capita (Millions)')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_per_capita_wealth_bar.png')
plt_cnt += 1

#-----------------------------------------------
# Plotting wealth based on normalized percentiles (just line graph)
#-----------------------------------------------
print(percentiles.values(), normalized_wealth.values)
plt.figure(figsize=(14, 8))
plt.plot(percentiles_str_list, normalized_wealth.values[::-1], color='red')
plt.title('Per-capita Wealth Distribution in 2020Q1 by Population Percentile')
plt.ylabel('Net Wealth per Person (Millions)')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_per_capita_wealth_line.png')
plt_cnt += 1
#-----------------------------------------------
# Interpolate
#-----------------------------------------------
# from scipy.interpolate import CubicSpline

# # Define the midpoints of each percentile range as the x-values
# # x_values = sorted([(start + end) / 2 for start, end in percentiles.values()])

# x_values = ([0.001/2 * 100,0.009/2 * 100,0.09/2 * 100,0.4/2 * 100,0.5/2 * 100])
# print(normalized_wealth[::-1])
# # Use the normalized_wealth values as the y-values
# y_values = normalized_wealth.values[::-1]

# # # Sort the x_values and y_values together based on x_values
# # sorted_indices = np.argsort(x_values)
# # x_values_sorted = np.array(x_values)[sorted_indices]
# # y_values_sorted = y_values[sorted_indices]

# # Perform cubic spline interpolation with the sorted values
# cs = CubicSpline(x_values, y_values)

# # Generate finer x-values for interpolation
# x_fine = np.linspace(0, 0.5/2 * 100, 500)
# y_fine = cs(x_fine)
# # Plotting the interpolated curve
# plt.figure(figsize=(14, 8))
# plt.plot(x_fine, y_fine, color='blue', label='Interpolated Wealth per Person')
# plt.scatter(x_values, y_values, color='red', s=100, zorder=5, label='Original Data Points')
# plt.title('Interpolated Wealth Distribution in 2020Q1 by Population Percentile')
# plt.ylabel('Wealth per Person (Millions)')
# plt.xlabel('Population Percentile')
# # plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
# plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.savefig('wealth_normalized_line_interpolated.png')

#-----------------------------------------------
# Scaled normalized wealth line graph
#-----------------------------------------------
plt.figure(figsize=(14, 8))

x_space = np.linspace(0, 100, 500)

print(normalized_wealth.values[::-1])
y_interp = np.interp(x_space, [size*100 for _, size in population_sizes.items()], normalized_wealth.values[::-1])
plt.plot(x_space, y_interp, color='red')
plt.title('Scaled Per-capita Wealth Distribution in 2020Q1 by Population Percentile')
plt.ylabel('Net Wealth per Person (Millions)')

# Adjust the y-axis to be in millions
locs, labels = plt.yticks()  # Get current y-axis tick locations and labels
plt.yticks(locs, [f"{x*1e-6:.1f}M" for x in locs])  # Set new labels in millions

plt.xlabel('Population Percentile')
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_scaled_wealth_normalized_line.png')
plt_cnt += 1

#-----------------------------------------------
# Scaled-log normalized wealth line graph
#-----------------------------------------------
plt.figure(figsize=(14, 8))

x_space = np.linspace(0, 100, 500)

# print(normalized_wealth.values[::-1])
y_interp = np.interp(x_space, [size*100 for _, size in population_sizes.items()], normalized_wealth.values[::-1])
plt.plot(x_space, y_interp, color='red')
plt.title('log-log Scaled Normalized Wealth Distribution in 2020Q1 by Population Percentile')
plt.ylabel('Net Wealth per Person (Millions)')
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
y_interp = np.interp(x_space, [size*100 for _, size in population_sizes.items()], normalized_wealth.values[::-1])

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
plt.ylabel('Net Wealth per Person (Millions)')
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
# y_interp = np.interp(x_space, [size*100 for _, size in population_sizes.items()], data_2020Q1.values[::-1])
# plt.plot(x_space, y_interp, color='red')
# plt.title('log-log Scaled Normalized Wealth Distribution in 2020Q1 by Population Percentile')
# plt.ylabel('Net Wealth per Person (Millions)')
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


#=============================================
# More granularity (new dataset)
#=============================================
net_worth_percentiles = pd.read_csv("net-worth-percentiles-2020-2023.csv", sep="\t")
# print(net_worth_percentiles)
# print([int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values][::-1])
# print([value.replace('$', '') for value in net_worth_percentiles['2023'].values])




#-----------------------------------------------
# Household wealth by percentile line graph 2020
#-----------------------------------------------

x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_percentiles['2020'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red')
plt.title('Household Wealth Distribution in 2020 by Population Percentile')
plt.ylabel('Net Wealth per Household')
plt.xlabel('Population Percentile')
# plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_scaled_household_wealth_2023_line.png')
plt_cnt += 1


#-----------------------------------------------
# log-log Household wealth by percentile line graph 2020
#-----------------------------------------------

# Problem: log-log doesn't handle negative values (1th percentile (or 99) has -100,000 in net worth)
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red')
plt.title('Household Wealth Distribution in 2020 by Population Percentile')
plt.ylabel('Net Wealth per Household')
plt.xlabel('Population Percentile')
plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xscale('log')
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_log-log_scaled_household_wealth_2023_line.png')
plt_cnt += 1

#-----------------------------------------------
# Household wealth by percentile line graph 2023
#-----------------------------------------------

x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_percentiles['2023'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red')
plt.title('Household Wealth Distribution in 2023 by Population Percentile')
plt.ylabel('Net Wealth per Household')
plt.xlabel('Population Percentile')
# plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_scaled_household_wealth_2020_line.png')
plt_cnt += 1


'''
	Let's consider the naive situation in which, within buckets
'''

#-----------------------------------------------
# log-log Household wealth by percentile line graph 2023
#-----------------------------------------------

# Problem: log-log doesn't handle negative values (1th percentile (or 99) has -100,000 in net worth)
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red', label='Normalized Wealth per Person')
plt.title('Household Wealth Distribution in 2023 by Population Percentile')
plt.ylabel('Net Wealth per Household')
plt.xlabel('Population Percentile')
plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xscale('log')
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_log-log_scaled_household_wealth_2023_line.png')
plt_cnt += 1

#-----------------------------------------------
# Household wealth by percentile line graph 2020 and 2023
#-----------------------------------------------

x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_percentiles['2020'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red', label='2020')
x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_percentiles['2023'].values][::-1])
plt.plot(x_space, y_interp, color='blue', label='2023')
plt.title('Household Wealth Distribution in 2020 and 2023 by Population Percentile')
plt.ylabel('Net Wealth per Household')
plt.xlabel('Population Percentile')
# plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_scaled_household_wealth_2020_and_2023_line.png')
plt_cnt += 1


#-----------------------------------------------
# log-log Household wealth by percentile line graph 2020 and 2023
#-----------------------------------------------

# Problem: log-log doesn't handle negative values (1th percentile (or 99) has -100,000 in net worth)
x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_percentiles['2020'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red', label='2020')
x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_percentiles['2023'].values][::-1])
plt.plot(x_space, y_interp, color='blue', label='2023')
plt.title('Household Wealth Distribution in 2020 and 2023 by Population Percentile')
plt.ylabel('Net Wealth per Household')
plt.xlabel('Population Percentile')
plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.xscale('log')
plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'out/{plt_cnt}_log-log_scaled_household_wealth_2020_and_2023_line.png')
plt_cnt += 1

#=============================================
# More granularity (new dataset)
#=============================================


#-----------------------------------------------
# Fitting pareto distribution
#-----------------------------------------------

# def generate_pareto_data(xm, alpha, size=1000):
#     return xm + np.random.pareto(alpha, size)

# # Define the Pareto Distribution Parameters
# xm = normalized_wealth.min()  # Set xm to the minimum value of normalized wealth
# alpha = 2  # Initial guess for alpha

# # Generate Pareto data for the relevant range
# pareto_data = {}
# for category, size in population_sizes.items():
#     pareto_data[category] = generate_pareto_data(xm, alpha, size=5)

# pareto_series = pd.Series(pareto_data)

# # Scale the Pareto data to fit the range of the graph
# # scaled_pareto_series = pareto_series * normalized_wealth.sum() / pareto_series.sum()

# # Plotting the normalized wealth and Pareto fit
# plt.figure(figsize=(14, 8))
# normalized_wealth.plot(color='red', label='Normalized Wealth per Person')
# pareto_series.plot(color='blue', label='Pareto Fit (alpha=2)')
# plt.title('Normalized Wealth Distribution in 2020Q1 by Population Percentile (Line Graph)')
# plt.ylabel('Net Wealth per Person (Millions)')
# plt.xlabel('Population Percentile')
# plt.xticks(rotation=45)
# plt.gca().invert_xaxis()
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
# plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.savefig('wealth_normalized_line_interpolated_pareto.png')

exit()



for t, d in zip(time_periods, data):
    sorted_data = sorted(d)  
    percentiles = np.linspace(0, 100, len(sorted_data))  # Create percentiles
    plt.plot(percentiles, sorted_data, label=f't={t}')

plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.title('Increase in Total Wealth Over Time with Pareto Distribution')
plt.xlabel('Percentile')
plt.ylabel('Wealth')
plt.legend()
plt.show()


from scipy.optimize import curve_fit
from scipy.stats import pareto

# Define the Pareto function for wealth as a function of percentile
def pareto_func(p, xm, alpha):
	return xm * (1/(1-p))**(1/alpha)

# Define the percentiles based on the provided categories
percentiles = {
	'TopPt1': 0.999,         # Top 0.1%
	'RemainingTop1': 0.99,  # Next 0.9%
	'Next9': 0.9,           # Next 9%
	'Next40': 0.5,          # Next 40% (taking it as the median of this group)
	'Bottom50': 0.25        # Bottom 50% (taking it as the median of this group)
}

# Initial guesses for the parameters
initial_guess_loc = 777847
initial_guess_scale = 2

# Fit the Pareto parameters for each time point
fitted_parameters = []
# https://stackoverflow.com/questions/3242326/fitting-a-pareto-distribution-with-python-scipy
for date, row in df_net_worth.iterrows():
	wealth_values = [row[category] for category in percentiles.keys()]
	p_values = list(percentiles.values())
	try:
		# params, _ = curve_fit(pareto_func, p_values, wealth_values, p0=initial_guess, maxfev=10000)
		params = pareto.fit((p_values, wealth_values), loc=initial_guess_loc, scale=initial_guess_scale)
		fitted_parameters.append({
			'Date': date,
			'xm': params[0],
			'alpha': params[1]
		})
	except Exception() as e:
		print(e)
		# Add NaN values for problematic data points
		fitted_parameters.append({
			'Date': date,
			'xm': float('nan'),
			'alpha': float('nan')
		})

# Convert the list of fitted parameters to a dataframe
df_parameters = pd.DataFrame(fitted_parameters)

print(df_parameters.head())

# Convert 'Date' from Period type to datetime type
df_parameters['Date'] = df_parameters['Date'].dt.to_timestamp()

# Re-plotting the evolution of xm and alpha over time

plt.figure(figsize=(14, 6))

# Plot for xm
plt.subplot(1, 2, 1)
plt.plot(df_parameters['Date'], df_parameters['xm'], '-o', label='xm (Scale Parameter)')
plt.title('Evolution of xm Over Time')
plt.xlabel('Date')
plt.ylabel('xm')
plt.xticks(rotation=45)

# Plot for alpha
plt.subplot(1, 2, 2)
plt.plot(df_parameters['Date'], df_parameters['alpha'], '-o', label='alpha (Shape Parameter)')
plt.title('Evolution of alpha Over Time')
plt.xlabel('Date')
plt.ylabel('alpha')
plt.yscale('log')  # Using a log scale due to the large range of alpha values
plt.xticks(rotation=45)

plt.show()

# Generate random data from Pareto distribution for given time t
def generate_pareto_data(xm, alpha, size=1000):
	return xm + np.random.pareto(alpha, size)

# Visualize the change over time
# time_periods = list(range(0, 50, 5))
# time_periods = [0, 5, 10]
time_periods = list(t for t in df_parameters['Date'])
data = [generate_pareto_data(xm, alpha) for xm, alpha in zip(df_parameters['xm'], df_parameters['alpha'])]

plt.figure(figsize=(12, 6))
for t, d in zip(time_periods, data):
	sorted_data = sorted(d)  
	percentiles = np.linspace(0, 100, len(sorted_data))  # Create percentiles
	plt.plot(percentiles, sorted_data, label=f't={t}')

plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.title('Increase in Total Wealth Over Time with Pareto Distribution')
plt.xlabel('Percentile')
plt.ylabel('Wealth')
plt.legend()
plt.show()

# # Pareto CDF function
# def pareto_cdf(x, xm=1, alpha=2):
#     return 1 - (xm / x)**alpha

# x_values = np.linspace(df_net_worth['Total Wealth'].min(), df_net_worth['Total Wealth'].max(), len(df_net_worth))

# xm_adjusted = df_net_worth['Total Wealth'].min()

# # Calculate the scaled Pareto CDF values
# pareto_cdf_values = pareto_cdf(x_values, xm=xm_adjusted)

# # Scale the Pareto CDF values to match the range of the total wealth values in the graph
# pareto_cdf_values_scaled = pareto_cdf_values * df_net_worth['Total Wealth'].max()