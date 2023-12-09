import numpy as np
from scipy.optimize import minimize
"""
Dagum  (1990,  1993,  1999,  2004) generalized his income and wealth distribution model specifying a model of net wealth
distribution with support x \in (-\infty, \infty) to account also for the high observed frequencies of  negative and null net wealth. 
Furthermore, it contains as particular cases Dagum Type I, eq. (56), and Type II, eq. (59). 

Unlike the right tails of income and (net and total) wealth distributions which present heavy tails, the left tail of net wealth distributions,  i.e.,  when the net wealth
tends to minus infinity, present a fast convergence to zero because of institutional and biological bounds to an unlimited increase
of the economic agents' liability.

The stylized facts outlined above determine the specification of a net wealth distribution model as a mixture of an atomic and 
two continuous distributions. The atomic distribution, F_2(x) concentrates its unit mass of economic agents at x=0. It accounts 
for the economic units with null income, net wealth and total wealth.  The continuous distribution F_1(x) accounts for the negative
net wealth observations. It has a fast left tail convergence to zero, hence, it has finite moments of all orders. 

The other continuous distribution, F_3(x) accounts for the positive values of income, net wealth and total wealth and presents
a heavy right tail, therefore, it ha  a small number of finite moments of order r>0. Sometimes, for net and total wealth, the variance
might become infinite. F_3(x) is specified as the Dagum Type I model.  

https://rivista-statistica.unibo.it/article/view/1243/667
"""

def x_neg(x):
	return np.minimum(x, 0)

def x_pos(x):
	return np.maximum(x, 0)

class DagumGeneralNetWealth():
	def __init__(self):
		pass

	def cdf(self, x, b1, b2, c, l, s, beta, delta):
		"""
		Cumulative distribution function of the Dagum Generalized Distribution.

		Parameters
		----------
		x : array_like
			quantiles
		b1, b2, b3, s, beta, delta	
			The shape parameter(s) for the distribution
		c, l:
			scale parameter

		Returns
		-------
		cdf : ndarray
			Cumulative distribution function evaluated at `x`
		""" 
		# Checks for parameters
		assert c > 0, "Parameter c must be greater than 0"
		assert s > 0, "Parameter s must be greater than 0"
		assert beta > 0, "Parameter beta must be greater than 0"
		assert l > 0, "Parameter l must be greater than 0"
		assert delta > 1, "Parameter delta must be greater than 1"

		alpha = b1 + b2
		b3 = 1 - alpha
		assert b2 >= 0, "Parameter b2 must be greater or equal to 0"
		assert b1 > 0 and b3 > 0, "Parameters b1, and b3 must be greater than 0"
		assert np.isclose(b1 + b2 + b3, 1), "The sum of b1, b2, and b3 must be equal to 1"
  
		x = np.asarray(x) 
  
		# with np.errstate(divide='ignore'):
		F1 = np.exp(-c * np.power(np.abs(x_neg(x)), s))
		F2 = np.maximum(x / np.abs(x), 0)
		F3 = np.power((1 + l * np.power(x_pos(x), -delta)), -beta)
  
		# F1 = np.where(
		# 	x <= 0,
		# 	np.exp(-c * np.power(np.abs(x), s)),
		# 	0
		# )
  
		# F2 = np.maximum(x / np.abs(x), 0)
  
		# F3 = np.where(
		# 	x >= 0,
		# 	np.power((1 + l * np.power(x, -delta)), -beta),
		# 	0
		# )
  
		return b1 * F1 + b2 * F2 + b3 * F3

	def pdf(self, x, b1, b2, c, l, s, beta, delta):
		"""Probability density function at x of the given RV.

		Parameters
		----------
		x : array_like
			quantiles
		arg1, arg2, arg3,... : array_like
			The shape parameter(s) for the distribution (see docstring of the
			instance object for more information)
		loc : array_like, optional
			location parameter (default=0)
		scale : array_like, optional
			scale parameter (default=1)

		Returns
		-------
		pdf : ndarray
			Probability density function evaluated at x

		"""
		# Checks for parameters
		assert c > 0, "Parameter c must be greater than 0"
		assert s > 0, "Parameter s must be greater than 0"
		assert beta > 0, "Parameter beta must be greater than 0"
		assert l > 0, "Parameter l must be greater than 0"
		assert delta > 1, "Parameter delta must be greater than 1"

		alpha = b1 + b2
		b3 = 1 - alpha
		assert b2 >= 0, "Parameter b2 must be greater or equal to 0"
		assert b1 > 0 and b3 > 0, "Parameters b1, and b3 must be greater than 0"
		assert np.isclose(b1 + b2 + b3, 1), "The sum of b1, b2, and b3 must be equal to 1"
  
		alpha = b1 + b2
		b3 = 1 - alpha
		x = np.asarray(x)



		f1 = np.where(
			x < 0,
			c * s * np.power(-x, s - 1) * np.exp(-c * np.power(-x, s)),
			0
		)
		f2 = np.where(
			x == 0,
			1,
			0
		)
		f3 = np.where(
			x > 0,
			beta * l * delta * np.power(x, beta * delta - 1) * np.power(np.power(x, delta) + l, -beta - 1),
			0
		)
  
		return b1 * f1 + b2 * f2 + b3 * f3

	def log_likelihood(self, params, x):
		b1, b2, c, l, s, beta, delta = params
		pdf_values = self.pdf(x, b1, b2, c, l, s, beta, delta)
		# To avoid log of zero
		pdf_values = np.where(pdf_values <= 0, np.finfo(float).eps, pdf_values)
		return -np.sum(np.log(pdf_values))

	def fit(self, x, initial_params):
		# Define constraints and bounds
		cons = ({'type': 'ineq', 'fun': lambda params: 1 - params[0] - params[1]})  # b3 = 1 - b1 - b2 > 0
		epsilon = 1e-6
		bounds = [(epsilon, 1), (0, 1), (epsilon, None), (epsilon, None), (epsilon, None), (epsilon, None), (1 + epsilon, None)]


		# Optimize
		result = minimize(self.log_likelihood, initial_params, args=(x,), method='trust-constr', bounds=bounds, constraints=cons, options={'disp': True})


		if result.success:
			fitted_params = result.x
		else:
			print(result.x)
			raise Exception('Optimization failed: ' + result.message)

		return fitted_params
  
if __name__=="__main__":
	dgnw = DagumGeneralNetWealth()
	# 1983 figures
	# in $10,000
	# -$100,000 to $1,000,000
	cdf = dgnw.cdf(np.linspace(-10, 100, 10000000), 0.0562, 0.00, 3.422, 463.85, 0.677, 0.207, 2.1823)
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
	# #================================================================
	# # Importing Data
	# #================================================================

	# psid_data = PSIDData()
	# # Equivalence scale adjusts net worth to individuals
	# equivalence_scale_adjust = False
	# psid_data.load(cpi_adjust=False, equivalence_scale_adjust=equivalence_scale_adjust, target_year=2019)
	# psid_wealth_dict = psid_data.get_household_wealth_data()


	# HOUSEHOLD = not equivalence_scale_adjust

	# #================================================================
	# # Picking out a single period
	# #================================================================

	# psid_chosen_period_df = psid_wealth_dict[PSID_CHOSEN_PERIOD]
	# psid_wealth_chosen_period_df = psid_chosen_period_df['IMP WEALTH W/ EQUITY']	
 
	# fit = dgnw.fit(psid_wealth_chosen_period_df, initial_params)
	# print(fit)
 
"""
	Return estimates of shape (if applicable), location, and scale
	parameters from data. The default estimation method is Maximum
	Likelihood Estimation (MLE), but Method of Moments (MM)
	is also available.

	Starting estimates for the fit are given by input arguments;
	for any arguments not provided with starting estimates,
	``self._fitstart(data)`` is called to generate such.

	One can hold some parameters fixed to specific values by passing in
	keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
	and ``floc`` and ``fscale`` (for location and scale parameters,
	respectively).

	Parameters
	----------
	data : array_like or `CensoredData` instance
		Data to use in estimating the distribution parameters.
	arg1, arg2, arg3,... : floats, optional
		Starting value(s) for any shape-characterizing arguments (those not
		provided will be determined by a call to ``_fitstart(data)``).
		No default value.
	**kwds : floats, optional
		- `loc`: initial guess of the distribution's location parameter.
		- `scale`: initial guess of the distribution's scale parameter.

		Special keyword arguments are recognized as holding certain
		parameters fixed:

		- f0...fn : hold respective shape parameters fixed.
			Alternatively, shape parameters to fix can be specified by name.
			For example, if ``self.shapes == "a, b"``, ``fa`` and ``fix_a``
			are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
			equivalent to ``f1``.

		- floc : hold location parameter fixed to specified value.

		- fscale : hold scale parameter fixed to specified value.

		- optimizer : The optimizer to use.  The optimizer must take
			``func`` and starting position as the first two arguments,
			plus ``args`` (for extra arguments to pass to the
			function to be optimized) and ``disp=0`` to suppress
			output as keyword arguments.

		- method : The method to use. The default is "MLE" (Maximum
			Likelihood Estimate); "MM" (Method of Moments)
			is also available.

	Raises
	------
	TypeError, ValueError
		If an input is invalid
	`~scipy.stats.FitError`
		If fitting fails or the fit produced would be invalid

	Returns
	-------
	parameter_tuple : tuple of floats
		Estimates for any shape parameters (if applicable), followed by
		those for location and scale. For most random variables, shape
		statistics will be returned, but there are exceptions (e.g.
		``norm``). 
"""

		# F1 = np.exp(-c * np.power(np.abs(x_neg(x)), s))
		# F2 = np.maximum(x / np.abs(x), 0)
		# F3 = np.power((1 + l * np.power(x_pos(x), -delta)), -bet