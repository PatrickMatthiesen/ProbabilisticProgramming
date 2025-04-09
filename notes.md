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

## week 4

Predictors that are correlated with each other can mask each other's effect on the outcome.
This will make the model worse.

### Causal inference

#### Fork: the classic hidden confounder variable

- condition on Z, make X and Y independent, even if they are dependent on a common cause
- the problem appears if we lack variables in the model; we don't condition on Z, and spurious dependence appears; __include Z__ to expose non-causal paths
- __the path from X and Y is blocked by conditioning on Z__
- Example: WaffleHouse-DivorceRate

#### Pipe: post-treatment bias

- condition on Z, X becomes independent of Y
- Appears when we have needless variables in the model, __remove Z__ to expose the causal path
- __the path from X to Y is blocked by conditioning on Z__
- Example: treatment-fungi-growth

#### Collider: X and Y are independent unless you condition on Z.

- the problem appears if we condition on a collider (perhaps unconsciously) and introduce a false dependency from Y or to X (or other variables)
- one has to be careful if the data we have is not conditioned on a collider already
- __the path from X to Y is open when conditioning on Z__; __remove Z__ to avoid the non-causal path
- Example: the modified fungi example, newsworthiness-trustworthiness, aga-marriage-happiness


#### Descendant: Any of the above indirectly

- Note that the descendant problem can be a variant of the fork, the pipe, or the collider pattern
- all these problems appear even if we don't condition on Z, but on a variable correlated with it (a descendant D)

### A procedure to address these 4 confounds, assuming you have a complete DAG

Given a causal DAG, it is always possible to say which, if any, variables one must control for in order to shut all the backdoor paths (non-causal paths), and which variables one must not control for, in order to avoid making new confounds. 

1. List all path connecting $X$ (the potential cause of interest) to $Y$ (the outcome of interest)
2. Classify each path by whether it is open or __closed (contains a collider)__
3. A path is a __backdoor__ if it has an arrow entering $X$
4. Decide which variables to condition on (to include in the model) or not to close the non-causal paths, using the patterns listed above, for forks, pipes, and colliders.

Because the set of rules is finite, there are libraries that can compute it for you, but it is always good to understand the problems yourself from the first principles.

The `CausalGraphicalModels` has two useful functions `is_valid_backdoor_adjustment_set` and `get_all_backdoor_adjustment_sets`. The first one checks if a set of variables is a valid backdoor adjustment set, and the second one returns all valid backdoor adjustment sets.


## week 5

**Agenda**
1. Overfitting and underfitting
2. Some (Information) Theory on accuracy of predictions
3. Curbing overfitting via regularization
4. Estimating accuracy of prediction (CV, PSIS, WAIC)
5. Prediction accuracy vs causality, robust regression

### Overfitting and underfitting

- Overfitting: the model is too complex, and it fits the noise in the data.
- Underfitting: the model is too simple, and it does not capture the underlying structure of the data.

#### How to detect overfitting

- Overfitting is detected by comparing the training and test error.
- If we train on a the data multiple times and leave different parts of the data set out. Then if the mean prediction function varies a lot, then the model is overfitting. The functions will be very different as the model will focus too much on the data that it is given. This is called cross-validation. 

### Information Theory

- Information theory is a branch of applied mathematics and electrical engineering involving the quantification of information.

#### Entropy

- Entropy is a measure of the uncertainty of a random variable.
- The entropy of a random variable is the average amount of information produced by a random variable.

Shannon's entropy is defined as:

$$
H(X) = -\sum_{i=1}^n p(x_i) \log_2 p(x_i)
$$

where $p(x_i)$ is the probability of the $i$-th outcome.

#### Regularization

- Regularization is a technique used to prevent overfitting.
- Regularization adds a penalty term to the loss function.

```python
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)
    mu = alpha + beta[0]*X1 + beta[1]*X2
    y = pm.Normal('y', mu=mu, sd=sigma, observed=Y)
    trace = pm.sample(2000, tune=1000)
```

```python
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)
    mu = alpha + beta[0]*X1 + beta[1]*X2
    y = pm.Normal('y', mu=mu, sd=sigma, observed=Y)
    trace = pm.sample(2000, tune=1000, nuts_kwargs={'target_accept': 0.95})
```

## week 6

Agenda: 
- Interactions with categorical regressors
- Interactions with continuous regressors

### Interactions

We have previously only used linear models, but we can also use interactions between variables.

for example, if we have a model like this:
M = d + b0*X0 + b1*X1

here we might be missing an interaction between X0 and X1. We can add this interaction by adding a new term to the model:

M = d + b0*f(X0, X1)
or maybe:
M = d + b0*X0 + b1*X1 + b2*f(X0, X1)

where f(X0, X1) is the interaction term.

### Interactions with categorical regressors

## week 7

### Beta-Binomial model

Beta is a conjugate prior for the binomial distribution. The beta-binomial model is a generalization of the binomial distribution, where the probability of success is not fixed, but follows a beta distribution.

The beta-binomial model is defined as:

$$
y \sim \mathrm{Binomial}(n, \theta)
$$

$$
\theta \sim \mathrm{Beta}(\alpha, \beta)
$$

where $\alpha$ and $\beta$ are the shape parameters of the beta distribution.

Beta prior can be a bad choice if the data is not binomial. For example, if the data is not binary, but continuous, the beta prior is not a good choice.

### Monte Carlo 

### Metrolpolis

### Gibbs

### Hamiltonian Monte Carlo


## week 8

### Big Entropy and the Generalized Linear Model

- The entropy of a distribution is a measure of the uncertainty of the distribution.
- The entropy of a distribution is the average amount of information produced by a random variable.

### Agenda

1. __Maximum Entropy__ Distributions
2. __Generalized Linear Model__ (the link functions)
3. __Summary__ of the regressions (time permitting)

### Entropy Definition

__Entropy__ (discrete and continuous, sometimes called differential entropy)

....

### Maximum Entropy Distributions

We want to use the distribution that has the maximum entropy, given the constraints we have.

The constraints are the moments of the distribution. The moments are the expected values of the distribution.

### Generalized Linear Model

We have previously used linear models, but we can also use generalized linear models.

The generalized linear model will be a non gaussian distribution with a function that maps us to our desired range.

For instance, if we wanted to optimize the probability of success in a Binomial distribution, we could use the following model:

$$ y_i \sim \mathrm{Binomial}(n, p_i) $$

$$ f(p_i) = \alpha + \beta(x_i - \overline x) $$

where $f(p_i)$ is the link function, and $\alpha + \beta(x_i - \overline x)$ is the linear predictor and $\overline x$ is the mean of the predictor.

## week 9

### Contrast

(good for the exam)

- Contrast is the difference between the expected value of the outcome variable for two different levels of a predictor variable.

### Poisson regression

- Poisson regression is a type of regression used for count data.
- Poisson regression is a generalized linear model with a log link function.
- Poisson regression is used when the outcome variable is a count variable, and the predictor variables are continuous or categorical.

### Binomial regression

- Binomial regression is a type of regression used for binary data.

## week 11

