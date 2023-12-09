import numpy as np
from scipy.optimize import minimize
from data import FedData, PSIDData
from utils.helper import calculate_percentiles
from utils.dagum_generalized import DagumGeneralNetWealth
from constants import PSID_CHOSEN_PERIOD

if __name__=="__main__":
	dgnw = DagumGeneralNetWealth()
	# 1983 figures
	# in $10,000
	# -$100,000 to $1,000,000
	# cdf = dgnw.cdf(np.linspace(-10, 100, 10000000), 0.0562, 0.00, 3.422, 463.85, 0.677, 0.207, 2.1823)
	import matplotlib.pyplot as plt
 
	# plt.figure(figsize=(14, 8))
	# plt.plot(np.linspace(-10, 100, 10000000), cdf)
	# plt.show()
 
	# pdf = dgnw.pdf(np.linspace(-10, 100, 100000), 0.0562, 0.00, 3.422, 463.85, 0.677, 0.207, 2.1823)
	# print(pdf)
	# plt.figure(figsize=(14, 8))
	# plt.plot(np.linspace(-10, 100, 100000), pdf)
	# plt.yscale('log')
	# plt.show() 
 
	initial_params = [0.0562, 0.00, 3.422, 463.85, 0.677, 0.207, 2.1823]
	# pdf = dgnw.pdf(np.linspace(-10, 100, 10_000_000,np.float64), 0.0562, 0.00, 3.422, 463.85, 0.677, 0.207, 2.1823)
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
 
	fit = dgnw.fit(psid_wealth_chosen_period_df, initial_params)
	print(fit)
 
	pdf = dgnw.pdf(np.linspace(-10, 100, 100000), *fit)
	print(pdf)
	plt.figure(figsize=(14, 8))
	plt.plot(np.linspace(-10, 100, 100000), pdf)
	plt.yscale('log')
	plt.show()  