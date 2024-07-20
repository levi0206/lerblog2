---
title: "Deep Learning Basics"
date: 2024-02-01T10:20:33+08:00
draft: false
ShowToc: true
---
<font color='green'>[Updated on 2024-07-18: New code exmaple.]</font>
<!-- <span style="color:MediumSeaGreen">[Updated on 2024-07-18: New code exmaple.]</span> -->


## Outline
In this post, we mention some important concepts in deep learning, including
- artificial neural network
- automatic differentiation
- Monte Carlo estimation 
- minibatch stochastic gradient descent,

These tools are fundamental in the modern deep-learning context. Given the extensive nature of each subsection, a comprehensive coverage is beyond the scope. Instead, we will pick certain subjects, explain some important ideas and theorems that support these mechanisms, and provide a toy example.


```python
# Standard PyTorch import
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Set seed
SEED = 2024 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Generate dataset
X, y = make_blobs(n_samples=1000,
                  n_features=2,
                  centers=2,
                  cluster_std=3,
                  random_state=1)
# Plot dataset
plt.scatter(x=X[:,0], y=X[0:,1], marker="o", c=y, s=25)
```    
![png](https://levi0206.github.io/lerblog2/basics/make_blobs.png)


```python
# Check shape of tensor
X = torch.Tensor(X)
y = torch.Tensor(y)
print("X shape:",X.shape)
print("y shape:",y.shape)
```

    X shape: torch.Size([1000, 2])
    y shape: torch.Size([1000])


## Artificial Neural Network

Artificial Neural Network, or abreviated neural network in machine learning context, is the core of deep learning. The name and structure are inspired by human brain, mimicking the interactions of biological neurons. 

Let 
- $x=(x_1,...,x_d)^T\in\mathbb{R}^d$ an input data vector
- $\sigma_i$ the $i$-th activation function such as ReLU, sigmoid or hyperbolic tangent that acts **component-wise** on a vector
- $W_i\in\mathbb{R}^{d_{i_1}\times d_{i_2}}$ the $i$-th weight matrix and $b_i\in\mathbb{R}^{d_{i_2}}$ the $i$-th bias vector. 

Let $A_i$ be the $i$-th affine map from $\mathbb{R}^{d_{{i-1}_2}}\to\mathbb{R}^{d_{{i}_2}}$
$$
A_i y = W_iy+b_i
$$_
where $y_i\in\mathbb{R}^{d_{i_1}}$ is the output of the previous layer.
A neural network of depth $k$ is a series of composition 
$$
\mathcal{N}\mathcal{N}(x) = \sigma_k \circ A_k \circ \cdots \circ \sigma_2 \circ A_2 \circ \sigma_1 \circ A_1 x.
$$


For example, a neural with single layer without activation can be implemented as


```python
# Don't execute
nn.Linear(input_dim,output_dim) 
```

which performs the linear transformations on $X_{n\times i}\in\mathbb{R}^{n\times i}$:
$$
Y_{n\times o} = X_{n\times i}W_{i\times o}+b_{o}
$$
where $W_{i\times o}$ is a weight matrix and $b_o$ is a bias vector. 

Similarly, we can implement a neural network using three layers with sigmoid for the first and second layer is like


```python
# Don't execute
nn.Sequential(
    nn.Linear(input_dim,hidden_1),
    nn.Sigmoid(),
    nn.Linear(hidden_1,hidden_2),
    nn.Sigmoid(),
    nn.Linear(hidden_2,output_dim)
)
```

In practice, one can implement the neural network with different tricks: adjusting the depth, number of layers, of neural network or width, number of neurons, of a layer, using dropout [1], using batch normalization [9], or even choosing another activation function if needed.


```python
class Classifier(nn.Module):
    def __init__(self,hidden=4):
        super(Classifier,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2,5),
            nn.Linear(5,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        output = self.model(x)
        return output
```

The universal approximation theorem is a key factor contributing to the widespread application of neural networks across various fields and scenarios. The universal approximation theorem originally is given by Cybenko in 1989 [6] using sigmoidal functions
$$
\sigma(x)=
\begin{cases}
1, & x\to \infty \\
0, & x\to-\infty
\end{cases}.
$$
Denote $I_n=[0,1]^n$ the $n$-dimensional unit cube and $C(I_n,\mathbb{R})$ space of continuous functions from $I_n$ to $\mathbb{R}$.

**Theorem: universal approximation theorem, G. Cybenko.** Let $\sigma$ be any continuous sigmoidal function, $f\in C(I_n,\mathbb{R})$. Then finite sums of the form
$$
G^N(x)=\sum_{j=1}^N \alpha_j\sigma(w_j^T x+\theta_j)
$$
are dense in the space of continuous functions $C(I_n,\mathbb{R})$. In other words, given any $f\in C(I_n,\mathbb{R})$, there exists $N\in\mathbb{N}$ such that
$$
|G^N(x)-f(x)|<\epsilon \quad \text{for all } x\in I_n.
$$
This original universal approximation theorem says a neural network with one layer and arbitrary width can approximate any continuous function on $[0,1]^n$, where $N$ is the number of neurons. Now there are many variants and generalizations of the theorem, for example [2],[8]. Another insight of the universality of neural network can be interpreted by ordinary differential equations, as referenced in [7].

## Monte Carlo Estimation
In a nutshell, Monte Carlo estimation simply says that one can replace an intractable integral by an average of summation.

Let $p(\mathbf{x},{\theta})$ be some probability distribution depending on a collection ${\theta}$ of parameters. Consider the expectation of the form 
$$
\mathcal{F}({\theta})=\int p(\mathbf{x},{\theta})f(\mathbf{x},{\phi})d\mathbf{x}=\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x},{\theta})}\left[f(\mathbf{x},{\phi})\right]
$$
where $\mathbf{x}$ is the input of objective $f$ with probability $p(\mathbf{x},{\theta})$, and ${\phi}$ is a set of the parameters of $f$. Of course, ${\phi}$ might be equal to ${\theta}$. We are interested in learning the parameters ${\theta}$, which requires the computation of the gradient of $\mathcal{F}({\theta})$ with respect to $\theta$:
$$
\nabla_{{\theta}}\mathcal{F}({\theta})=\nabla_{{\theta}}\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x},{\theta})}\left[f(\mathbf{x},{\phi})\right].
$$
The expectation in general is intractable because the distribution $p(\mathbf{x},{\theta})$ might be high-dimensional, in deep learning, easily in the order of thousands or even more of dimensions, and very complicated. Moreover, the function $f$ might be non-differentiable, or a black-box function which the output is all we observe, artificial neural network for example. 


The Monte Carlo Method provides another insight of this sort of impossible calculation. Instead of computing the closed form of the integral directly, we draw i.i.d. samples $\hat{\mathbf{x}}^{(1)},...,\hat{\mathbf{x}}^{(S)}$ from $p(\mathbf{x},{\theta})$, and approximate the integral with the average of $f(\hat{\mathbf{x}}^{(i)},{\phi})$, called a Monte Carlo estimator:
$$
\bar{\mathcal{F}}_S=\frac{1}{S}\sum_{i=1}^S f(\hat{\mathbf{x}}^{(i)},{\phi}).
$$
Although $\bar{\mathcal{F}}_S$ is still a random variable because it depends on random variables $\hat{\mathbf{x}}^{(1)},...,\hat{\mathbf{x}}^{(S)}$, now it is equiped with desirable properties:

**Unbiasedness.**
$$
\begin{align*}
\mathbb{E}_{\hat{\mathbf{x}}^{(i)}\sim p(\hat{\mathbf{x}}^{(i)},{\theta})}\left[\bar{\mathcal{F}}_S\right] & = \mathbb{E}_{\hat{\mathbf{x}}^{(i)}\sim p(\hat{\mathbf{x}}^{(i)},{\theta})}\left[\frac{1}{S}\sum_{i=1}^S f(\hat{\mathbf{x}}^{(i)},{\phi})\right] = \frac{1}{S}\sum_{i=1}^S\mathbb{E}_{\hat{\mathbf{x}}^{(i)}\sim p(\hat{\mathbf{x}}^{(i)},{\theta})}\left[ f(\hat{\mathbf{x}}^{(i)},{\phi})\right] \\
& = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x},{\theta})}\left[f(\mathbf{x},{\phi})\right].
\end{align*}
$$
Unbiasedness is always preferred because it allows us to guarantee the convergence of a stochastic optimisation procedure.

**Consistency.** 
By strong law of large numbers, the random variable $\bar{\mathcal{F}}_S$ converges to $\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x},{\theta})}\left[f(\mathbf{x},{\phi})\right]$ almost surely as the number of samples $S$ increases.

Monte Carlo estimation provides a convenient approach to approximate expectations. For example, in generative adversarial networks, the discriminator is updated with the loss function
$$
-\mathbb{E}_{\hat{\mathbf{x}}\sim p_{data}}\left[\log D(\hat{\mathbf{x}})\right]-\mathbb{E}_{\hat{\mathbf{z}}\sim \mathcal{N}(0,\mathbf{I})} \left[\log (1-D(G(\hat{\mathbf{z}})))\right].
$$
where $D$ is the discriminator network, $G$ is the generator network, $\hat{\mathbf{x}}^{(i)}$ is a datapoint sampled from data distribution $p_{data} $ and $z$ is a random noise vector sampled from $\mathcal{N}(0,\mathbf{I})$. With Monte Carlo estimation, instead of calculating two intractable integrals, we only need to calculate the average
$$
-\frac{1}{m}\sum_{i=1}^m \ln D(\hat{\mathbf{x}}^{(i)})-\frac{1}{m}\sum_{i=1}^m \ln(1-D(G(\hat{\mathbf{z}}^{(i)}))).
$$
The quantity $m$, or batch size, is a hyperparameter. 

In our classification task, if the output probability of a datapoint is $\geq 0.5$, it's identified as the class with label "1" otherwise "0". Thus, we use **binary cross entropy** as our loss function, 
$$
-\mathbb{E}_{\hat{\mathbf{x}}^{(i)}\sim y_i=1} \log p_i-\mathbb{E}_{\hat{\mathbf{x}}^{(i)}\sim y_i=0} \log (1-p_i).
$$
where $p_i=\text{Classifier}(\hat{\mathbf{x}}^{(i)})$ and $y_i$ is the label of $\hat{\mathbf{x}}^{(i)}$. With Monte Carlo estimation, we can implement the loss function
$$
-\left[\frac{1}{m}\sum_{i=1}^{m} y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$
with full batch $m=256$ in our case.


```python
criterion = nn.BCELoss()
```

## Automatic Differentiation
Automatic differentiation is widely used for deep learning optimization. It is a clever way of performing chain rule without manually computing derivatives. If you use PyTorch, you don't have to implement autdifferentiation yourself. You can perform autdifferentiation with `autograd` in few lines.

Let $f=f_u\circ f_{u-1}\circ \cdots \circ f_1$ be a function. Each $f_i:\mathbb{R}^n\to\mathbb{R}$ is differentiable. Chain rule of differentiation tells us how to compute the derivative:
$$
\frac{df}{dx} = \frac{df_u}{df_{u-1}}\cdots\frac{df_{2}}{df_{1}}\frac{df_1}{dx}.
$$
**Forward-mode.** Forward-mode autodifferentiation performs the chain rule in the fashion
$$
\frac{df_i}{dx} = \frac{df_i}{df_{i-1}}\frac{df_{i-1}}{dx} \quad \text{for $i=2,...,u$},
$$
or
$$
\begin{equation}
\frac{df}{dx} = \frac{df_u}{df_{u-1}}\left(\cdots\left(\frac{df_3}{df_2}\left(\frac{df_2}{df_1}\frac{df_1}{dx}\right)\right)\right) \tag{F1}.
\end{equation}
$$
**Reverse-mode.** Reverse-mode, also known as backpropagation, performs the chain rule in another way
$$
\frac{df_u}{df_{i-1}} = \frac{df_u}{df_{i}}\frac{df_{i}}{df_{i-1}} \quad \text{for $i=u-1,...,1$ with $f_0=x$},
$$
namely
$$
\begin{equation}
\frac{df}{dx} = \left(\left(\left(\frac{df_u}{df_{u-1}}\frac{df_{u-1}}{df_{u-2}}\right)\frac{df_{u-2}}{df_{u-3}}\right)\cdots\right)\frac{df_1}{dx} \tag{R1}. 
\end{equation}
$$
#### Computational Efficiency
Each evaluation of $\text{(F1)}$ autodifferentiation is a matrix-matrix product, while each calculation of $\text{(R1)}$ is just a vector-matrix product. Thus, the reverse-mode autodifferentiation, or backpropagation, is always preferred for training neural networks.
#### What Autodifferentiation is not
Autodifferentiation is a **procedure of calculating derivatives**; it does not aim to find the closed form of solution. 

**It is not**
- **finite differences**
$$
\frac{\partial}{\partial x_i}f(x_1,...,x_N) \approx \frac{f(x_1,...,x_i+h,...,x_N)-f(x_1,...,x_i,...,x_N)}{h}
$$
which are expensive and induce numerical error. We only use finite differences for testing the gradient. 
- **symbolic differentiation** which can result in complex and redundant expressions.

## Minibatch Stochastic Gradient Descent
Nowadays, people usually prefer minibatch stochastic gradient descent to optimize neural networks, as it is more computationally efficient than gradient descent or stochastic gradient descent. 

Let $\ell$ be our loss function. Let $n$ be the number of training datapoint, $b$ the batch size, ${\theta^0}\in\mathbb{R}^d$ the initial parameter of our neural network and $(\alpha_t)_{t\in\mathbb{N}}$ a sequence of step size, or learning rate. Given a subsest $B\subset \{1,...,n\}$, we define
$$
\nabla_{{\theta^t}} \ell_B({\theta^t})=\frac{1}{|B|}\sum_{i\in B} \nabla_{{\theta^t}} \ell_i({\theta^t})
$$
The minibatch stochastic gradient descent algorithm is given by 
$$
\begin{aligned}
& B_t\subset\{1,...,n\} \quad\quad \text{Sampled uniformly among sets of size $b$} \\
& {\theta^{t+1}} = {\theta^t}-\alpha_t \nabla_{{\theta^t}} \ell_B({\theta^t}).
\end{aligned}
$$
Note that $\nabla_{{\theta^t}} \ell_B({\theta^t})$ is an unbiased estimator
$$
\mathbb{E}_{b}\left[\nabla_{{\theta^t}} \ell_B({\theta^t})\right] = \frac{1}{{n \choose b}} \sum_{\substack{B\subset\{1,...,n\} \\ |B|=b}} \nabla_{{\theta^t}} \ell_B({\theta^t}) = \nabla_{{\theta^t}} \ell({\theta^t})
$$
because each batch is sampled with probability $\frac{1}{{n \choose b}}$. Readers interested in the convergence of minibatch SGD with convex and smooth functions may check [3] for related theorems and proofs.



```python
n_epoch = 1000
lr = 0.0005
model = Classifier()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss_record = []
acc_record = []

for i in range(n_epoch):
    model.train()
    
    output_prob = model(X).squeeze()
    pred = (output_prob >= 0.5).float()

    # Calculate loss
    loss = criterion(output_prob, y)
    loss_record.append(loss.item())

    # Calculate accuracy
    correct = (pred == y).sum().item()
    acc = correct / len(pred) * 100
    acc_record.append(acc)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%100==0:
        print("Epoch {} loss {:.4f} acc {:.2f}%".format(i+1, loss, acc))
```
Training log:
```
    Epoch 100 loss 0.6060 acc 62.00%
    Epoch 200 loss 0.4755 acc 69.70%
    Epoch 300 loss 0.3889 acc 78.20%
    Epoch 400 loss 0.3318 acc 83.90%
    Epoch 500 loss 0.2934 acc 87.40%
    Epoch 600 loss 0.2667 acc 90.50%
    Epoch 700 loss 0.2474 acc 91.70%
    Epoch 800 loss 0.2329 acc 92.40%
    Epoch 900 loss 0.2216 acc 92.50%
    Epoch 1000 loss 0.2125 acc 92.90%
```
The performance of the model gradually increases as the figure shown below.
```python
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(loss_record, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(acc_record, label='Training Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
```
    
![png](https://levi0206.github.io/lerblog2/basics/loss_acc.png)

## Conclusion
In this post, we discussed several key concepts in deep learning, including artificial neural networks, automatic differentiation, Monte Carlo estimation, and minibatch stochastic gradient descent. Each of these components is crucial for developing and optimizing neural network models.

- **Artificial Neural Networks** form the backbone of deep learning, capable of approximating complex functions, supported by the Universal Approximation Theorem.
- **Automatic Differentiation** is essential for efficiently computing gradients, fundamental for training neural networks through optimization algorithms like gradient descent.
- **Monte Carlo Estimation** provides a method for approximating intractable integrals, useful for training models whose loss function includes the calculation of expectations.
- **Minibatch Stochastic Gradient Descent** balances computational efficiency and convergence speed, as demonstrated in our classifier training example.

Hope the readers unedrstand these essential ingredients in almost every deep-learning models.

## References
[1] Srivastava, Nitish, Geoffrey E, Hinton, Alex, Krizhevsky, Ilya, Sutskever, Ruslan, Salakhutdinov. "Dropout: a simple way to prevent neural networks from overfitting". Journal of machine learning research 15. 1(2014): 1929–1958.

[2] Zhou, Ding-Xuan. "Universality of Deep Convolutional Neural Networks.". CoRR abs/1805.10769. (2018).

[3] Bubeck, Sébastien. "Convex Optimization: Algorithms and Complexity.". Foundations and Trends in Machine Learning 8. 3-4(2015): 231–357.

[4] Baydin, Atilim Gunes, Barak A., Pearlmutter, Alexey Andreyevich, Radul, Jeffrey Mark, Siskind. "Automatic Differentiation in Machine Learning: a Survey.". J. Mach. Learn. Res. 18. (2017): 153:1–153:43.

[5] Mohamed, Shakir, Mihaela, Rosca, Michael, Figurnov, Andriy, Mnih. "Monte Carlo Gradient Estimation in Machine Learning.". J. Mach. Learn. Res. 21. (2020): 132:1–132:62.

[6] Cybenko, George. "Approximation by superpositions of a sigmoidal function.". Math. Control. Signals Syst. 2. 4(1989): 303–314.

[7] Kidger, Patrick. "On Neural Differential Equations." (2022). 

[8] Pinkus, Allan. "Approximation theory of the MLP model in neural networks". Acta Numerica 8. (1999): 143–195.

[9] Ioffe, Sergey, Christian, Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." Proceedings of the 32nd International Conference on Machine Learning. PMLR, 2015.

[10] Ruder, Sebastian. "An overview of gradient descent optimization algorithms." (2016). 

