---
title: "Understand Automatic Differentiation"
date: 2024-07-17T00:00:42+08:00
draft: false
---

Automatic differentiation, or AD, is crucial in deep learning and widely used in almost every neural network optimization because it enables the efficient and accurate computation of gradients, which are essential for training models through techniques such as gradient descent. It has been integrated into many deep-learning frameworks such as PyTorch and TensorFlow, allowing users to perform AD on neural networks with just a few lines of code. This post aims to clarify concepts such as forward mode, reverse mode, and computational graphs, though from an engineering perspective, it is still possible to build models without a deep understanding of automatic differentiation.

## Outline
- Forward mode and reverse mode of automatic differentiation
- Computational graph illustration

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
in which $|\mathbf{x}|$ denotes the dimension of $\mathbf{x}$ and the size of each Jacobian matrix is annoted using underbraces. The "forward" and "reverse" refer to the **order** of calculating derivatives. In the forward mode, we calculate $\frac{df}{d\mathbf{x}}$ in this fashion
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