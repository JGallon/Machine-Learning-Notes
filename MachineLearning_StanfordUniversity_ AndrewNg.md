# Supervised Learning

- Regression problem
- Classification problem

# Unsupervised Learning

- Clustering problem
- Cocktail party problem

# Linear Regression with One Variable

**Notation**

m = Number of training examples

x = "input" variable / feature

y = "output" variable / "target" variable

$(x, y)$ = one training example

$(x^{(i)}, y^{(i)})$ = $i^{th}$ training example

**Hypothesis**

$h_\theta(x) = \theta_0 + \theta_1 \times x$

**Cost function**

$J(\theta_0, \theta_1) = \frac{1}{2 \times m} \times \sum_{i=1}^m{(h_\theta(x^{(i)}) - y^{(i)})^2}$

假设 $\theta_0 = 0$

$J(0, \theta_1) = \frac{1}{2 \times m} \times \sum_{i=1}^m{(\theta_1 \times x^{(i)} - y^{(i)})^2}$

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}{J(\theta_0, \theta_1)}\quad for(j = 0\ and\ j = 1)$

$\alpha \longrightarrow learning\ rate$

**Correct Simultaneous update**

$temp0 := \theta_0 - \alpha \frac{\partial}{\partial\theta_0}{J(\theta_0, \theta_1)}$

$temp1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1}{J(\theta_0, \theta_1)}$

$\theta_0 := temp0$

$\theta_1 := temp1$