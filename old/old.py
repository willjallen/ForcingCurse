#=============================================
# More granularity (new dataset)
#=============================================
net_worth_fed_data.percentiles = pd.read_csv("net-worth-fed_data.percentiles-2020-2023.csv", sep="\t")
# print(net_worth_fed_data.percentiles)
# print([int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values][::-1])
# print([value.replace('$', '') for value in net_worth_fed_data.percentiles['2023'].values])




#-----------------------------------------------
# Household wealth by percentile line graph 2020
#-----------------------------------------------

x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_fed_data.percentiles['2020'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red')
plt.title('Household Wealth Distribution in 2020 by Population Percentile')
plt.ylabel('Net Worth per Household')
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
plt.ylabel('Net Worth per Household')
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
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_fed_data.percentiles['2023'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red')
plt.title('Household Wealth Distribution in 2023 by Population Percentile')
plt.ylabel('Net Worth per Household')
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
plt.ylabel('Net Worth per Household')
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
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_fed_data.percentiles['2020'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red', label='2020')
x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_fed_data.percentiles['2023'].values][::-1])
plt.plot(x_space, y_interp, color='blue', label='2023')
plt.title('Household Wealth Distribution in 2020 and 2023 by Population Percentile')
plt.ylabel('Net Worth per Household')
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
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_fed_data.percentiles['2020'].values][::-1])
plt.figure(figsize=(14, 8))
plt.plot(x_space, y_interp, color='red', label='2020')
x_space = np.linspace(0, 100, 500)
y_interp = np.interp(x_space, [int(size.replace('%', '')) for size in net_worth_fed_data.percentiles['Percentile'].values], [float(value.replace('$', '').replace(',', '')) for value in net_worth_fed_data.percentiles['2023'].values][::-1])
plt.plot(x_space, y_interp, color='blue', label='2023')
plt.title('Household Wealth Distribution in 2020 and 2023 by Population Percentile')
plt.ylabel('Net Worth per Household')
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
# for category, size in fed_data.population_sizes.items():
#     pareto_data[category] = generate_pareto_data(xm, alpha, size=5)

# pareto_series = pd.Series(pareto_data)

# # Scale the Pareto data to fit the range of the graph
# # scaled_pareto_series = pareto_series * normalized_wealth.sum() / pareto_series.sum()

# # Plotting the normalized wealth and Pareto fit
# plt.figure(figsize=(14, 8))
# normalized_wealth.plot(color='red', label='Normalized Wealth per Person')
# pareto_series.plot(color='blue', label='Pareto Fit (alpha=2)')
# plt.title('Normalized Wealth Distribution in 2020Q1 by Population Percentile (Line Graph)')
# plt.ylabel('Net Worth per Person (Millions)')
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
    fed_data.percentiles = np.linspace(0, 100, len(sorted_data))  # Create fed_data.percentiles
    plt.plot(fed_data.percentiles, sorted_data, label=f't={t}')

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

# Define the fed_data.percentiles based on the provided categories
fed_data.percentiles = {
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
	wealth_values = [row[category] for category in fed_data.percentiles.keys()]
	p_values = list(fed_data.percentiles.values())
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
	fed_data.percentiles = np.linspace(0, 100, len(sorted_data))  # Create fed_data.percentiles
	plt.plot(fed_data.percentiles, sorted_data, label=f't={t}')

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


#-----------------------------------------------
# Interpolate
#-----------------------------------------------
# from scipy.interpolate import CubicSpline

# # Define the midpoints of each percentile range as the x-values
# # x_values = sorted([(start + end) / 2 for start, end in fed_data.percentiles.values()])

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
