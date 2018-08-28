# Introduction

**Supervised Learning**

- Regression problem
- Classification problem

**Unsupervised Learning**

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

**Gradient descent algorithm**

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}{J(\theta_0, \theta_1)}\quad for(j = 0\ and\ j = 1)$

$\alpha \longrightarrow learning\ rate$

**Correct Simultaneous update**

$temp0 := \theta_0 - \alpha \frac{\partial}{\partial\theta_0}{J(\theta_0, \theta_1)}$

$temp1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1}{J(\theta_0, \theta_1)}$

$\theta_0 := temp0$

$\theta_1 := temp1$

$\frac{\partial}{\partial \theta_j}{J(\theta_0, \theta_1)} = \frac{\partial}{\partial \theta_j} \times \frac{1}{2 \times m} \times \sum_{i=1}^m{(h_\theta(x^{(i)}) - y^{(i)})^2}$

$\frac{\partial}{\partial \theta_j}{J(\theta_0, \theta_1)} = \frac{\partial}{\partial \theta_j} \times \frac{1}{2 \times m} \times \sum_{i=1}^m{(\theta_0 + \theta_1 \times x^{(i)} - y^{(i)})^2}$

$j = 0: \frac{\partial}{\partial \theta_0}{J(\theta_0, \theta_1)} = \frac{1}{m} \times \sum_{i=1}^m{(h_\theta(x^{(i)}) - y^{(i)})}$

$j = 1: \frac{\partial}{\partial \theta_1}{J(\theta_0, \theta_1)} = \frac{1}{m} \times \sum_{i=1}^m{(h_\theta(x^{(i)}) - y^{(i)}) \times x^{(i)}}$

$temp0 := \theta_0 - \alpha \times \frac{1}{m} \times \sum_{i=1}^m{(h_\theta(x^{(i)}) - y^{(i)})}$

$temp1 := \theta_1 - \alpha \times \frac{1}{m} \times \sum_{i=1}^m{(h_\theta(x^{(i)}) - y^{(i)}) \times x^{(i)}}$

$\theta_0 := temp0$

$\theta_1 := temp1$

**"Batch" Gradient Descent**

"Batch": Each step of gradient descent uses all the training examples.

# Linear Algebra Review

**Matrix Elements (entries of matrix)**

$A = \begin{bmatrix}
1402&191\\1371&821\\949&1437\\147&1448\\
\end{bmatrix}$

$A_{ij} =$ "$i, j$ entry" in the $i^{th}$ row, $j^{th}$ column.

**Vector:** An n x  1 matrix.

$y = \begin{bmatrix}
460\\232\\315\\178\\
\end{bmatrix}$

$y_i = i^{th}$ element

**Matrix Addition**

$\begin{bmatrix}1&0\\2&5\\3&1\end{bmatrix} + \begin{bmatrix}4&0.5\\2&5\\0&1\end{bmatrix} = \begin{bmatrix}5&0.5\\4&10\\3&2\end{bmatrix}$

**Scalar Multiplication**

$3 \times \begin{bmatrix}1&0\\2&5\\3&1\end{bmatrix} = \begin{bmatrix}3&0\\6&15\\9&3\end{bmatrix}$

$\begin{bmatrix}4&0\\6&3\end{bmatrix} \div 4 = \begin{bmatrix}1&0\\1.5&0.75\end{bmatrix}$

**Combination of Operands**

$3 \times \begin{bmatrix}1\\4\\2\end{bmatrix} + \begin{bmatrix}0\\0\\5\end{bmatrix} - \begin{bmatrix}3\\0\\2\end{bmatrix} \div 3$

**Matrix Vector Multiplication**

$A \times x = y$

$A \longrightarrow$ m $\times$ n matrix ( m rows, n columns )

$x \longrightarrow$ n $\times$ 1 matrix ( n-dimensional vector )

$y \longrightarrow$ m $\times$ 1 matrix ( m-dimensional vector )

To get $y_i$, multiply A's $i^{th}$ row with elements of vector x, and add them up.

**Matrix Matrix Multiplication**

$A \times B = C$

$A \longrightarrow$ m $\times$ n matrix ( m rows, n columns )

$B \longrightarrow$ n $\times$ o matrix ( m rows, o columns )

$C \longrightarrow$ m $\times$ o matrix

The $i^{th}$ column of the matrix $C$ is obtained by multiplying $A$ with the $i^{th}$ column of $B$. ( for $i$ = 1, 2, ...., o )

| House sizes |
| - | 
| 2104 | 
| 1416 | 
| 1534 |
| 852 |


| Have 3 competing hypotheses |
| - |
| 1. $h_\theta(x) = -40 + 0.25x$ |
| 2. $h_\theta(x) = 200 + 0.1x$ |
| 3. $h_\theta(x) = -150 + 0.4x$ |

$\begin{bmatrix}1&2014\\1&1416\\1&1534\\1&852\end{bmatrix} \times \begin{bmatrix}-40&200&-150\\0.25&0.1&0.4\end{bmatrix} = \begin{bmatrix}486&410&692\\314&342&416\\344&353&464\\173&285&191\end{bmatrix}$

**Matrix Multiplication Properties**

Let $A$ and $B$ be matrices. Then in general, $A \times B \neq B \times A$. ( not commutative. )

$A \times B \times C = A \times ( B \times C)$

**Identity Matrix**

Denoted $I$ ( or $I_{n \times n}$ )

Examples of identity matrices:

$\begin{bmatrix}
 1      & 0      & \cdots & 0      \\
 0      & 1      & \cdots & 0      \\
 \vdots & \vdots & \ddots & \vdots \\
 0      & 0      & \cdots & 1      \\
\end{bmatrix}$

For any matrix $A$,

$A \times I = I \times A = A$

**Matrix inverse**

if $A$ is an $m \times m$ matrix, and if it has an inverse.

$AA^{-1} = A^{-1}A = I$

Matrices that don't have an inverse are "singular" or "degenerate"

**Matrix Transpose**

$A = \begin{bmatrix}
    1 & 2 & 0 \\
    3 & 5 & 9
\end{bmatrix}$&nbsp;&nbsp;&nbsp;&nbsp;
$A^T = \begin{bmatrix}
    1 & 3 \\
    2 & 5 \\
    0 & 9
\end{bmatrix}$

# Linear Regression with Multiple Variables

**Notation**

$n =$ number of features

$x^{(i)} =$ input (features) of $i^{th}$ training example

$x^{(i)}_j =$ value of feature $j$ in $i^{th}$ training example

**Hypothesis**

previously: $h_\theta(x) = \theta_0 + \theta_1x$

$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n$

for convenience of notation, define $x_0 = 1.$

$x = \begin{bmatrix}
    x_0\\
    x_1\\
    x_2\\
    \vdots\\
    x_n
\end{bmatrix}$&nbsp;&nbsp;&nbsp;&nbsp;
$\theta = \begin{bmatrix}
   \theta_0\\
   \theta_1\\
   \theta_2\\
   \vdots\\
   \theta_n\\ 
\end{bmatrix}$

$h_\theta(x) = \theta^T \times x$

**Gradient descent for multipe variables**

Hypothesis: $h_\theta(x) = \theta^Tx = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n\ \ \ \ (x_0 = 1)$

Parameters: $\theta_0, \theta_1, \cdots, \theta_n$

Cost function:

$J(\theta_0, \theta_1, \cdots, \theta_n) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2$

**Gradient descent:**

$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta_0, \cdots, \theta_n)$ simultaneously update for every $j = 0, \cdots, n$

New algorithm $(n \geq 1)$

$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$ simultaneously update $\theta_j$ for $j = 0,\cdots, n$

