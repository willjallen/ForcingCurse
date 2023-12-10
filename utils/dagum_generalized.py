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
        Calculates the Cumulative Distribution Function (CDF) of the Dagum Generalized Distribution.

        Parameters
        ----------
        x : array_like
            Array of quantiles at which to evaluate the CDF.
        b1, b2 : float
            Shape parameters of the distribution. The sum of b1, b2, and (1 - b1 - b2) should be 1.
        c : float
            Scale parameter for the negative part of the distribution (x < 0). Must be > 0.
        l : float
            Scale parameter for the positive part of the distribution (x > 0). Must be > 0.
        s : float
            Shape parameter for the negative part of the distribution. Must be > 0.
        beta : float
            Shape parameter for the positive part of the distribution. Must be > 0.
        delta : float
            Additional shape parameter for the positive part of the distribution. Must be > 1.

        Returns
        -------
        ndarray
            Cumulative distribution function values evaluated at `x`.
        """
		# Checks for parameters
		assert c > 0, f"Parameter c must be greater than 0, current value: {c}"
		assert s > 0, f"Parameter s must be greater than 0, current value: {s}"
		assert beta > 0, f"Parameter beta must be greater than 0, current value: {beta}"
		assert l > 0, f"Parameter l must be greater than 0, current value: {l}"
		assert delta > 1, f"Parameter delta must be greater than 1, current value: {delta}"

		alpha = b1 + b2
		b3 = 1 - alpha
		assert b2 >= 0, f"Parameter b2 must be greater or equal to 0, current value: {b2}"
		assert b1 > 0 and b3 > 0, f"Parameters b1 and b3 must be greater than 0, current values: b1={b1}, b2={b2}, b3={b3}"
		assert np.isclose(b1 + b2 + b3, 1), f"The sum of b1, b2, and b3 must be equal to 1, current sum: {b1 + b2 + b3}"
  
		x = np.asarray(x) 
  
		F1 = np.zeros_like(x) 
		neg_mask = x < 0
		F1[neg_mask] = np.exp(-c * np.power(np.abs(x[neg_mask]), s))
		
		F2 = np.maximum(x / np.abs(x), 0)
  
		F3 = np.zeros_like(x)
		pos_mask = x > 0
		F3[pos_mask] = np.power((1 + l * np.power(x[pos_mask], -delta)), -beta)

		return b1 * F1 + b2 * F2 + b3 * F3

	def pdf(self, x, b1, b2, c, l, s, beta, delta):
		"""
        Calculates the Probability Density Function (PDF) of the Dagum Generalized Distribution.

        Parameters
        ----------
        x : array_like
            Array of quantiles at which to evaluate the PDF.
        b1, b2 : float
            Shape parameters of the distribution. The sum of b1, b2, and (1 - b1 - b2) should be 1.
        c : float
            Scale parameter for the negative part of the distribution (x < 0). Must be > 0.
        l : float
            Scale parameter for the positive part of the distribution (x > 0). Must be > 0.
        s : float
            Shape parameter for the negative part of the distribution. Must be > 0.
        beta : float
            Shape parameter for the positive part of the distribution. Must be > 0.
        delta : float
            Additional shape parameter for the positive part of the distribution. Must be > 1.

        Returns
        -------
        ndarray
            Probability density function values evaluated at `x`.
        """
		# Checks for parameters
		assert c > 0, f"Parameter c must be greater than 0, current value: {c}"
		assert s > 0, f"Parameter s must be greater than 0, current value: {s}"
		assert beta > 0, f"Parameter beta must be greater than 0, current value: {beta}"
		assert l > 0, f"Parameter l must be greater than 0, current value: {l}"
		assert delta > 1, f"Parameter delta must be greater than 1, current value: {delta}"

		alpha = b1 + b2
		b3 = 1 - alpha
		assert b2 >= 0, f"Parameter b2 must be greater or equal to 0, current value: {b2}"
		assert b1 > 0 and b3 > 0, f"Parameters b1 and b3 must be greater than 0, current values: b1={b1}, b2={b2}, b3={b3}"
		assert np.isclose(b1 + b2 + b3, 1), f"The sum of b1, b2, and b3 must be equal to 1, current sum: {b1 + b2 + b3}"
  
		alpha = b1 + b2
		b3 = 1 - alpha
		x = np.asarray(x)

		# Numpy np.where is a conditional on choosing from existing arrays, not a condition for their creation, thus producing invalid values when producing said arrays.
		f1 = np.zeros_like(x) 
		neg_mask = x < 0
		f1[neg_mask] = c * s * np.power(-x[neg_mask], s - 1) * np.exp(-c * np.power(-x[neg_mask], s))
  
		f2 = np.where(
			x == 0,
			1,
			0
		)
  
		f3 = np.zeros_like(x)
		pos_mask = x > 0
		f3[pos_mask] = beta * l * delta * np.power(x[pos_mask], beta * delta - 1) * np.power(np.power(x[pos_mask], delta) + l, -beta - 1)
  
		return b1 * f1 + b2 * f2 + b3 * f3


	def log_likelihood(self, params, x):
		"""
        Computes the negative log-likelihood of the given data under the Dagum Generalized Distribution.

        Parameters
        ----------
        params : tuple of float
            Parameters of the distribution (b1, b2, c, l, s, beta, delta).
        x : array_like
            Array of data points for which to compute the log-likelihood.

        Returns
        -------
        float
            Negative log-likelihood of the given data under the distribution.
        """ 
		b1, b2, c, l, s, beta, delta = params
		pdf_values = self.pdf(x, b1, b2, c, l, s, beta, delta)
		# To avoid log of zero
		pdf_values = np.where(pdf_values <= 0, np.finfo(float).eps, pdf_values)
		return -np.sum(np.log(pdf_values))

	def fit(self, x, initial_params):
		"""
        Fits the Dagum Generalized Distribution to a given dataset by optimizing the distribution parameters.

        Parameters
        ----------
        x : array_like
            Array of data points to fit the distribution to.
        initial_params : tuple of float
            Initial guess for the distribution parameters (b1, b2, c, l, s, beta, delta).

        Returns
        -------
        ndarray
            Optimized distribution parameters.

        Raises
        ------
        Exception
            If optimization fails, an exception is raised with the failure message.
        """ 
		# Define constraints and bounds
		epsilon = 1e-9
		bounds = [(epsilon, 1), (0, 1), (epsilon, 100), (epsilon, None), (epsilon, 100), (epsilon, 100), (1 + epsilon, 100)]
		cons = (
				{'type': 'ineq', 'fun': lambda params: 1 - params[0] - params[1]}  # b3 = 1 - b1 - b2 > 0
			)

		# Optimize
		result = minimize(self.log_likelihood, initial_params, args=(x,), method='trust-constr', constraints=cons, bounds=bounds, options={'disp': True, 'maxiter': 10_000}, tol=1e-6)

		if result.success:
			fitted_params = result.x
		else:
			print(result.x)
			raise Exception('Optimization failed: ' + result.message)

		return fitted_params