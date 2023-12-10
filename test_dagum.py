import numpy as np
from scipy.optimize import minimize
from data import FedData, PSIDData
from utils.helper import calculate_percentiles
from utils.dagum_generalized import DagumGeneralNetWealth
from constants import PSID_CHOSEN_PERIOD
import matplotlib.pyplot as plt

if __name__=="__main__":
	dgnw = DagumGeneralNetWealth()
	# 1983 figures
	# in $10,000
	# -$100,000 to $1,000,000
	# cdf = dgnw.cdf(np.linspace(-10, 100, 100000), 0.0562, 0.09, 3.422, 463.85, 0.677, 0.207, 2.1823)
 
	# plt.figure(figsize=(14, 8))
	# plt.plot(np.linspace(-10, 100, 100000), cdf)
	# plt.show()
 
	# pdf = dgnw.pdf(np.linspace(-10, 100, 100000), 0.0562, 0.00, 3.422, 463.85, 0.677, 0.207, 2.1823)
	# print(pdf)
	# plt.figure(figsize=(14, 8))
	# plt.plot(np.linspace(-10, 100, 100000), pdf)
	# plt.yscale('log')
	# plt.show() 
# The maximum number of function evaluations is exceeded.
# Number of iterations: 100000, function evaluations: 800160, CG iterations: 597549, optimality: 6.78e-03, constraint violation: 0.00e+00, execution time: 1.2e+03 s.
# [2.03783141e-01 8.73657266e-02 9.90708208e+00 8.81972201e+03
#  6.84661991e-01 2.48950254e-02 8.96540606e+00] 
	initial_params = [0.0562, 0.9, 3.422, 9463.85, 0.677, 9.807, 9.1823]
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
	psid_wealth_chosen_period_df = psid_chosen_period_df['IMP WEALTH W/ EQUITY']/1_000_000
 
	fit = dgnw.fit(psid_wealth_chosen_period_df, initial_params)
	print(fit)
 
	space = np.linspace(np.min(psid_wealth_chosen_period_df), np.max(psid_chosen_period_df), 1000000)
 
	pdf = dgnw.pdf(space, *fit)
	plt.figure(figsize=(14, 8))
	plt.plot(space, pdf)
	plt.yscale('log')
	plt.xscale('symlog')
	plt.show()  
 
	cdf = dgnw.cdf(space, *fit)
	plt.figure(figsize=(14, 8))
	plt.plot(space, cdf)
	plt.yscale('log')
	plt.xscale('symlog')
	plt.show()   