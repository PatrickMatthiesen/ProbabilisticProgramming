# week 2

## 4E1

> In the model definition below, which line is the likelihood?

```
yi ∼ Normal(µ, σ)
µ ∼ Normal(0, 10)
σ ∼ Exponential(1)
```

The likelihood is the first line, `yi ∼ Normal(µ, σ)`. This is the line that specifies the distribution of the data, given the parameters.

## 4E2

> In the model definition just above, how many parameters are in the posterior distribution?

The posterior distribution has two parameters: µ and σ. These are the parameters that we are trying to estimate.
We can see it by looking at the model definition: `µ ∼ Normal(0, 10)` and `σ ∼ Exponential(1)`.
These are used in the likelihood to define the posterior distribution.

## 4E3

> Using the model definition above, write down the appropriate form of Bayes’ theorem that includes the proper likelihood and priors.

Unmodified Bayes' theorem is:
$$
p(θ|y) = \frac{p(y|θ)p(θ)}{p(y)}
$$
where:

- $p(θ|y)$ is the posterior distribution
- $p(y|θ)$ is the likelihood
- $p(θ)$ is the prior distribution
- $p(y)$ is the marginal likelihood
- $y$ is the data

In the model definition above, the likelihood is `yi ∼ Normal(µ, σ)`, the prior for µ is `µ ∼ Normal(0, 10)`, and the prior for σ is `σ ∼ Exponential(1)`. The appropriate form of Bayes' theorem is:
$$
p(µ, σ|y) = \frac{p(y|µ, σ)p(µ)p(σ)}{p(y)}
$$

What we did was to replace the general symbols with the specific ones from the model definition.
We replaced:
- $θ$ with $(µ, σ)$, because $(µ, σ)$ are the priors
- $p(θ|y)$ with $p(µ, σ|y)$
- $p(y|θ)$ with $p(y|µ, σ)$
- $p(θ)$ with $p(µ)p(σ)$, because $p(µ)p(σ)$ is the same as $p(µ, σ)$
- $θ$ with $(µ, σ)$

## 4E4

> In the model definition below, which line is the linear model?

```
yi ∼ Normal(µ, σ)
µi = α + βxi
α ∼ Normal(0, 10)
β ∼ Normal(0, 1)
σ ∼ Exponential(2)
```

The linear model is the second line, `µi = α + βxi`. This is the line that specifies the relationship between the data and the parameters.

## 4E5

In the model definition just above, how many parameters are in the posterior distribution?

We have 3 parameters in the posterior distribution: α, β, and σ. These are the parameters that we are trying to estimate.

## 4M1

> For the model definition below, simulate observed y values from the prior (not the posterior).

```
yi ∼ Normal(µ, σ)
µ ∼ Normal(0, 10)
σ ∼ Exponential(1)
```

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 100
mu = np.random.normal(0, 10)
sigma = np.random.exponential(1)
y = np.random.normal(mu, sigma, n)

plt.hist(y, bins=30)
plt.show()
```

## 4M2

> Translate the model just above into a quap formula.

## 4M4

> A sample of students is measured for height each year for 3 years. After the third year, you want to fit a linear regression predicting height using year as a predictor. Write down the mathematical model definition for this regression, using any variable names and priors you choose. Be prepared to defend your choice of priors.

The model definition is:

```
yi ∼ Normal(µ, σ)
µi = α + βxi
α ∼ Normal(130, 30) - in case the mean height is 160 cm
β ∼ LogNormal(0, 1) - in case the slope is 0
σ ∼ Exponential(2)
```

where:
- $y_i$ is the height of student $i$
- $µ_i$ is the expected height of student $i$
- $α$ is the intercept
- $β$ is the slope
- $x_i$ is the year of measurement for student $i$
- $σ$ is the standard deviation of the height
- $µ$ is the mean height

The priors are:
- $α ∼ Normal(130, 30)$: The choice of prior is based on the assumption that the mean height is around 160 cm, with a standard deviation of 30.
- $β ∼ LogNormal(0, 1)$: The choice of prior is based on the assumption that the slope is positive.
- $σ ∼ Exponential(2)$: The choice of prior is based on the assumption that the standard deviation of the height is exponentially distributed with a rate parameter of 2.

The choice of priors is based on the assumption that the intercept and slope are normally distributed around 0, with a standard deviation of 10 and 1, respectively. The standard deviation of the height is assumed to be exponentially distributed with a rate parameter of 2.

## 4M5

> Now suppose I remind you that every student got taller each year. Does this information lead you to change your choice of priors? How?


## 4M6

> Now suppose I tell you that the variance among heights for students of the same age is never
more than 64cm. How does this lead you to revise your priors?

Because the variance among heights for students of the same age is never more than 64cm, we can revise the prior for the standard deviation of the height to be less than 8 (the square root of 64). This would lead to a more informative prior for the standard deviation of the height, which would help to constrain the estimates of the parameters.
