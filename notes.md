# Probabilistic programming notes

## week 3

- If we have more than one predictor, it is important to know how they interact with each other. Thus the correlation between the predictors is important.
- counterfactuals are statements about what would have happened if the world had been different.
- Masked Interactions: When the effect of one predictor is masked by another predictor.
  - For example if we have two predictors, and one of them is a strong predictor, the other predictor might not be significant.

### Exponential distribution (density function)

$$
f(x|\lambda) = \lambda e^{-\lambda x}
$$

where $x \geq 0$ and $\lambda > 0$.
The mean of the exponential distribution is $1/\lambda$ and the variance is $1/\lambda^2$.

- $\lambda$ is the rate parameter of the distribution.
- $\lambda = 1/\mu$ where $\mu$ is the mean of the distribution.
- $\lambda = 1/\sigma$ where $\sigma$ is the standard deviation of the distribution.
- The __maximum entropy distribution__ for a random variable which is non-negative and has a fixed expectation:
  - Contains minimum information besides the expected displacement from zero.
  - It fixes the expectation and allows many values that are smaller, and larger.
- $\sigma \sim \mathrm{Exponential}(1)$ means we expect the standard deviation to be 1, and now nothing more about it.


### counterfactuals

- Counterfactuals are statements about what would have happened if the world had been different.
- Allowing for counterfactuals is a key feature of Bayesian statistics.
- Allows us to understand a predictors effect on the outcome.
- Counterfactuals are not observable, but we can make inferences about them.

## random stuff

- Normal distribution allows for some noise
- using different amount of points makes it easier to understand errors.
- always do a prior predictive check to see if the model is working correctly.
  - verry important to do this.
- Understanding the standard deviation of the data is important.
  - The standard deviation of the data is the amount it fluctuates around the mean.
