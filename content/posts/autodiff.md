---
title: "Understand Automatic Differentiation"
date: 2024-07-17T00:00:42+08:00
draft: false
---

Automatic differentiation, or AD, is crucial in deep learning and widely used in almost every neural network optimization because it enables the efficient and accurate computation of gradients, which are essential for training models through techniques such as gradient descent. It has been integrated into many deep-learning frameworks such as PyTorch and TensorFlow, allowing users to perform AD on neural networks with just a few lines of code. This post aims to clarify concepts such as forward mode, reverse mode, and computational graphs, though from an engineering perspective, it is still possible to build models without a deep understanding of automatic differentiation.

## Outline
- Numerical differentiation and symbolic differentiation
- Automatic differentiation
- Computational graph illustration

## Numerical differentiation and symbolic differentiation
It must be clarified that the automatic differentiation is **not numerical differentiation**, which calculates the derivative of $f$ using definition
$$
\frac{d}{dx}f(x) \approx \frac{f(x+h)-f(x)}{h}, \quad h \text{ small},
$$
which requires numerical precision. Due to the nature of the method, the calculation of the derivative will definitely cause error that we do not want to expect. Also, the accuracy is directly influenced by the choice of the stepsize $h$. Consider a function
$$
g(x) = \exp((-3)\cos^2(4x))
$$
with derivative
$$
\frac{dg}{dx} = 24*\exp((-3)\cos^2(4x))*\cos(4x)*\sin(4x).
$$
Now we calculate the exact derivative and the estimated derivative using foward-difference method:
```python
def g(x):
    return x + np.exp(-3 * np.cos(4 * x)**2)

# Exact derivative
def dgdx(x):
    return 1 + 24 * np.exp(-3 * np.cos(4 * x)**2) * np.cos(4 * x) * np.sin(4 * x)

# Forward difference with stepsize h
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

x = np.linspace(0, 2, 1000)
hs = np.logspace(-13, 1, 1000)

errs = np.zeros(len(hs))

for i, h in enumerate(hs):
    # Compuate the difference and store L2 norm of error
    err = forward_diff(g, x, h) - dgdx(x) 
    errs[i] = np.linalg.norm(err) 

# Plot
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(hs, errs, lw=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('h', fontsize=12)
ax.set_ylabel(r'$\|g^{\prime}_{FD}-g^{\prime}_{exact}\|_{L_2}$', fontsize=14)
plt.tight_layout()
```
The $L_2$ norm error of $g_{exact}'$ and $g_{FD}'$ with the change of stepsize is given by
![png](https://levi0206.github.io/lerblog2/autodiff/finite_diff.png)

The choice of approximation scheme also affects the error.
```python
def g(x):
    return x + np.exp(-3 * np.cos(4 * x)**2)

def dgdx(x):
    return 1 + 24 * np.exp(-3 * np.cos(4 * x)**2) * np.cos(4 * x) * np.sin(4 * x)

def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def centered_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

x = np.linspace(0, 2, 1000)
hs = np.logspace(-13, 1, 1000)

ferrs = np.zeros(len(hs))
berrs = np.zeros(len(hs))
cerrs = np.zeros(len(hs))

for i, h in enumerate(hs):
    ferr = forward_diff(g, x, h) - dgdx(x)  
    ferrs[i] = np.linalg.norm(ferr)  # store L2 norm of error
    berr = backward_diff(g, x, h) - dgdx(x)  
    berrs[i] = np.linalg.norm(berr)  # store L2 norm of error
    cerr = centered_diff(g, x, h) - dgdx(x)  
    cerrs[i] = np.linalg.norm(cerr)  # store L2 norm of error

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(hs, ferrs, label='Forward Difference')
ax[0].plot(hs, cerrs, label='Centered Difference')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('h', fontsize=12)
ax[0].set_ylabel(r'$\|g^{\prime}_{FD}-g^{\prime}_{exact}\|_{L_2}$', fontsize=14)
ax[0].tick_params(labelsize=12)
ax[0].legend()

ax[1].plot(hs, berrs, label='Backward Difference')
ax[1].plot(hs, cerrs, label='Centered Difference')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('h', fontsize=12)
ax[1].set_ylabel(r'$\|g^{\prime}_{BD}-g^{\prime}_{exact}\|_{L_2}$', fontsize=14)
ax[1].tick_params(labelsize=12)
ax[1].legend()

plt.tight_layout()
plt.show()
```
![png](https://levi0206.github.io/lerblog2/autodiff/finite_diff_compare.png)
## Automatic Differentiation
Let first consider a composition function $f$ without specifying its domain and co-domain
$$
f(x) = f_n(f_{n-1}(\ldots f_1(x)))
$$
and suppose for every $i$, $f_i$ is differentiable. By chain rule, we know that the derivative of $f$ evaluated at $x_0$ is given by 
$$
\frac{df}{dx}(x_0) = \frac{df_n}{df_{n-1}}(x_0)\cdots\frac{df_2}{df_1}(x_0)\frac{df_1}{dx}(x_0).
$$
For example, let $f:\mathbb{R}\to\mathbb{R}$ defined by
$$
f(x) = e^{-x^2} = \exp(-x^2).
$$
Write $y=y(x)=-x^2$; the derivative $f'$ is given by
$$
\begin{aligned}
f'(x) & = \frac{df}{dy}\frac{dy}{dx} \\
& = \frac{d}{dy} e^y \cdot \frac{d}{dx} (-x^2) \\
& = e^{y(x)} \cdot (-2x) \\
& = e^{-x^2} \cdot (-2x).
\end{aligned}
$$
Now we may consider a differentiable function $f:\mathbb{R}^n\to\mathbb{R}^{n_3}$ with vector input. Let 
$$
f = f_3 \circ f_2 \circ f_1 
$$
where
- $f_1:\mathbb{R}^n\to\mathbb{R}^{n_1}$
- $f_2:\mathbb{R}^{n_1}\to\mathbb{R}^{n_2}$
- $f_3:\mathbb{R}^{n_2}\to\mathbb{R}^{n_3}$

are differentiable, $n,n_1,n_2,n_3\in\mathbb{N}$. For a vector $\mathbf{x}\in\mathbb{R}^n$, write 
$$
\mathbf{a} = f_1(\mathbf{x}), \quad \mathbf{b} = f_2(\mathbf{a}), \quad \mathbf{c} = f_3(\mathbf{b})
$$
where the bold letter denotes a vector. The Jacobian matrix of $f$ expressed as
$$
\frac{df}{d\mathbf{x}} = 
\left[
\begin{matrix}
\frac{df_1}{dx_1} & \dots & \frac{df_1}{dx_n} \\
\vdots & \ddots & \vdots \\
\frac{df_{n_3}}{dx_1} & \dots & \frac{df_{n_3}}{dx_n}
\end{matrix}
\right] \in\mathbb{R}^{n_3\times n}
$$
can be obtained by a product of Jacobian matrices
$$
\underbrace{\frac{df}{d \mathbf{x}}}_{|\mathbf{c}|\times|\mathbf{x}|} = \underbrace{\frac{d \mathbf{c}(\mathbf{b})}{d \mathbf{b}}}_{|\mathbf{c}|\times|\mathbf{b}|} \underbrace{\frac{d \mathbf{b}(\mathbf{a})}{d \mathbf{a}}}_{|\mathbf{b}|\times|\mathbf{a}|} \underbrace{\frac{d \mathbf{a}(\mathbf{x})}{d \mathbf{x}}}_{|\mathbf{a}|\times|\mathbf{x}|}
$$
in which $|\mathbf{x}|$ denotes the dimension of $\mathbf{x}$ and the size of each Jacobian matrix is annoted using underbraces. Instead of using $n,n_1,n_2,n_3$, the notation $|\mathbf{x}|$ makes more clear in the following discussion. The "forward" and "reverse" refer to the **order** of calculating derivatives. In the forward mode, we calculate $\frac{df}{d\mathbf{x}}$ in this fashion
$$
\frac{df}{d\mathbf{x}} = \frac{d\mathbf{c}}{d\mathbf{b}}\left(\frac{d\mathbf{b}}{d\mathbf{a}}\frac{d\mathbf{a}}{d\mathbf{x}}\right), \\
$$
and in the reverse mode, we compute
$$
\frac{df}{d\mathbf{x}} = \left(\frac{d\mathbf{c}}{d\mathbf{b}}\frac{d\mathbf{b}}{d\mathbf{a}}\right)\frac{d\mathbf{a}}{d\mathbf{x}}.
$$
Now, note that for two matrices $A,B$ of size $a\times b$ and $b\times c$, the number of multiplications in the product $A\cdot B$ is $a\cdot b\cdot c$. Thus, the forward mode requires
$$
|\mathbf{x}|\cdot|\mathbf{a}|\cdot|\mathbf{b}|+|\mathbf{x}|\cdot|\mathbf{b}|\cdot|\mathbf{c}| = |\mathbf{x}|\cdot|\mathbf{b}|(|\mathbf{a}|+|\mathbf{c}|),
$$
while the reverse mode requires
$$
|\mathbf{c}|\cdot|\mathbf{a}|\cdot|\mathbf{b}|+|\mathbf{c}|\cdot|\mathbf{a}|\cdot|\mathbf{x}| = |\mathbf{c}|\cdot|\mathbf{a}|(|\mathbf{b}|+|\mathbf{x}|).
$$
Which one is more efficient? It depends, but let's first consider a simple case $|\mathbf{c}|\geq|\mathbf{b}|\geq|\mathbf{a}|\geq|\mathbf{x}|$; namely, the input is low dimensional and the output is high dimensional. In this case, the forward mode is more efficient because
$$
\frac{|\mathbf{c}|\cdot|\mathbf{a}|}{|\mathbf{x}|\cdot|\mathbf{b}|}\ge\frac{|\mathbf{c}|+|\mathbf{a}|}{|\mathbf{x}|+|\mathbf{b}|}.
$$
Conversely, if $|\mathbf{c}|\leq|\mathbf{b}|\leq|\mathbf{a}|\leq|\mathbf{x}|$. For instance, $f$ is vector-input-scalar-output; then the reverse mode is more efficient. However, there's an intuitive way of seeing this. Let $\mathbf{x}$ be a vector and let $f$ output a scalar. Suppose the output of $f_i$ is a vevtor. Then the forward-mode AD performs matrix-matrix products, while the reverse mode performs vector-matrix products, much more efficient than the forward mode, which gives the reason why in practice, we always use reverse-mode AD because the loss function is typically a real-valued function taking a vector as input.