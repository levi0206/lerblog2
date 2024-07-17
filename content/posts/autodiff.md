---
title: "Understand Automatic Differentiation"
date: 2024-07-17T00:00:42+08:00
draft: false
---

Automatic differentiation, or AD, is crucial in deep learning and widely used in almost every neural network optimization because it enables the efficient and accurate computation of gradients, which are essential for training models through techniques such as gradient descent. It has been integrated into many deep-learning frameworks such as PyTorch and TensorFlow, allowing users to perform AD on neural networks with just a few lines of code. This post aims to clarify concepts such as forward mode, reverse mode, and computational graphs, though from an engineering perspective, it is still possible to build models without a deep understanding of automatic differentiation.
- Forward mode and reverse mode of automatic differentiation
- Computational graph illustration

Let first consider a composition function $f$ without specifying its domain and co-domain
$$
f(x) = f_n(f_{n-1}(\ldots f_1(x)))
$$
and suppose for every $i$, $f_i$ is differentiable. From calculus, we know that the derivative of $f$ evaluated at $x_0$ is given by 
$$
\frac{df}{dx}(x_0) = \frac{df_n}{df_{n-1}}(x_0)\cdots\frac{df_2}{df_1}(x_0)\frac{df_1}{dx}(x_0).
$$
For example, let
$$
f(x) = e^{-x^2} = \exp(x^2).
$$
Write $y=y(x)=x^2$; the derivative $f'$ is given by
$$
\begin{aligned}
f'(x) & = \frac{df}{dy}\frac{dy}{dx} \\
& = \frac{d}{dy} e^y \cdot \frac{d}{dx} -x^2 \\
& = e^{y(x)} \cdot (-2x) \\
& = e^{-x^2} \cdot (-2x).
\end{aligned}
$$