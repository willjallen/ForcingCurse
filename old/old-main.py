import numpy as np
import matplotlib.pyplot as plt

# Time-dependent scale and shape parameters
def xm_t(t, W_0=1, r=0.03):  # assuming a 3% growth rate as an example
    return W_0 * np.exp(r * t)


# Empirical studies on wealth distribution in various countries have found that the Pareto index α typically falls in the range of 1.5 to 3 for many real-world economies.
def alpha_t(t):
    return 2 - t/100 # Keeping this constant for simplicity, but you can adjust if desired

# Generate random data from Pareto distribution for given time t
def generate_pareto_data(t, size=1000):
    xm = xm_t(t)
    alpha = alpha_t(t)
    return xm + np.random.pareto(alpha, size)

# Visualize the change over time
time_periods = list(range(0, 50, 5))
# time_periods = [0, 5, 10]
data = [generate_pareto_data(t) for t in time_periods]

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






# Calculate the CDF for given data
def calculate_cdf(data):
    sorted_data = np.sort(data)
    return sorted_data, np.arange(1, len(data)+1) / len(data)

# Calculate the PDF for given data
def calculate_pdf(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist





# # Visualize the change over time
# time_periods = list(range(0, 50, 5))
# data = [generate_pareto_data(t) for t in time_periods]

# plt.figure(figsize=(12, 6))
# for t, d in zip(time_periods, data):
#     x, y = calculate_cdf(d)
#     plt.plot(x, y, label=f't={t}')

# plt.title('Evolution of Wealth Distribution Over Time (CDF)')
# plt.xlabel('Wealth')
# plt.ylabel('CDF')
# plt.legend()
# plt.grid(True)
# plt.show()