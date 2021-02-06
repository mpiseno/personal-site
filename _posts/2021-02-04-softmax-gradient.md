---
layout: post
title: Gradient of Softmax Cross-Entropy Loss
date: 2021-02-4 11:12:00-0400
description:
released: true
---

There's been a considerable number of students who have trouble deriving the gradient of the softmax cross-entropy loss function and translating it into code. In this post, I'll derive it step-by-step so that you all have an easier time translating it to code, but I will not provide code because that's part of the homework.

## Notation

We will denote the input to the softmax function (i.e. the logits) $$\mathbf{z}$$ and the softmax function $$S(\mathbf{z})$$ which produces a vector output $$\mathbf{s} = S(\mathbf{z})$$. Let there be $$k$$ classes so that $$\mathbf{z, s} \in \mathbb{R}^{k}$$ and denote the index of the correct class $$c$$. Recall that the softmax function produces a probability distribution over the classes. We denote the true distribution of the classes as $$\mathbf{y}$$, which is a one-hot vector where index $$c$$ is set to 1 (i.e. $$\mathbf{y}_{c} = 1$$ and $$\mathbf{y}_{i}$$ = 0 for $$i \neq c$$).

Let $$L(\mathbf{s})$$ be the cross-entropy loss function defined as

$$
L(\mathbf{s}, \mathbf{y}) = -\sum_{i}\mathbf{y}_{i} \log \mathbf{s}_{i}
$$


## Computing the Gradient

First let's simplify the cross-entropy loss function by using the fact that $$\mathbf{y}$$ is a one-hot vector.

$$
\begin{align*}
    L(\mathbf{s}, \mathbf{y}) &= -\sum_{i}\mathbf{y}_{i} \log \mathbf{s}_{i} \\
    &= -\log \mathbf{s}_{c}
\end{align*}
$$

Now we can use the chain rule to note that $$\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{s}_{c}}\frac{\partial \mathbf{s}_{c}}{\partial \mathbf{z}}$$. $$\frac{\partial L}{\partial \mathbf{s}_{c}}$$ is just $$\frac{-1}{\mathbf{s}_{c}}$$. $$\frac{\partial \mathbf{s}_{c}}{\partial \mathbf{z}}$$ is slightly more complicated. We will compute it by considering the $$i$$th entry - that is $$\frac{\partial \mathbf{s}_{c}}{\partial \mathbf{z}_{i}}$$.

If $$i = c$$ then we have

$$
\begin{align*}
    \frac{\partial \mathbf{s}_{c}}{\partial \mathbf{z}_{i}} &= \frac{\partial}{\partial \mathbf{z}_{i}} \frac{e^{\mathbf{z}_{c}}}{\sum_{j}e^{\mathbf{z}_{j}}} \\
    &= \frac{e^{\mathbf{z}_{c}}\sum_{j}e^{\mathbf{z}_{j}} - e^{\mathbf{z}_{c}}e^{\mathbf{z}_{c}}}{(\sum_{j}e^{\mathbf{z}_{j}})^{2}} \\
    &= \frac{e^{\mathbf{z}_{c}}}{\sum_{j}e^{\mathbf{z}_{j}}}\frac{\sum_{j}e^{\mathbf{z}_{j}} - e^{\mathbf{z}_{c}}}{\sum_{j}e^{\mathbf{z}_{j}}} \\
    &= \mathbf{s}_{c}(1 - \mathbf{s}_{c})
\end{align*}
$$

If $$i \neq c$$, we get

$$
\begin{align*}
    \frac{\partial \mathbf{s}_{c}}{\partial \mathbf{z}_{i}} &= \frac{\partial}{\partial \mathbf{z}_{i}} \frac{e^{\mathbf{z}_{c}}}{\sum_{j}e^{\mathbf{z}_{j}}} \\
    &= \frac{-e^{\mathbf{z}_{i}}e^{\mathbf{z}_{c}}}{(\sum_{j}e^{\mathbf{z}_{j}})^{2}} \\
    &= -\mathbf{s}_{i}\mathbf{s}_{c}
\end{align*}
$$

Now we can return to computing the full gradient $$\frac{\partial L}{\partial \mathbf{z}}$$.

$$
\begin{align*}
    \frac{\partial L}{\partial \mathbf{z}} &= \frac{\partial L}{\partial \mathbf{s}_{c}}\frac{\partial \mathbf{s}_{c}}{\partial \mathbf{z}} \\
    &= \frac{-1}{\mathbf{s}_{c}}
    \begin{bmatrix} -\mathbf{s}_{1}\mathbf{s}_{c} & -\mathbf{s}_{2}\mathbf{s}_{c} & ... & \mathbf{s}_{c}(1 - \mathbf{s}_{c}) & ... & -\mathbf{s}_{k}\mathbf{s}_{c} \end{bmatrix} \\
    &= \begin{bmatrix} \mathbf{s}_{1} & \mathbf{s}_{2} & ... & (\mathbf{s}_{c} - 1) & ... & \mathbf{s}_{k} \end{bmatrix}
\end{align*}
$$

So the gradient of the softmax cross-entropy amounts to subtracting 1 from value at the index of the correct class! We also could have computed the full row vector $$\frac{\partial L}{\partial \mathbf{s}}$$ and Jacobian matrix $$\frac{\partial \mathbf{s}}{\partial \mathbf{z}}$$ and matrix multiplied them. However, the former is sparse, so it simplified the computation a lot to do it how we did above. Furthermore, the method above shows that when computing gradients for backpropagation, we can use the fact that the flow of computation is such that the entries of $$\mathbf{s}$$ that are not at the correct class's index do not contribute to the loss, so we don't need to consider them.

Hopefully this helped you guys out. Stay positive and test negative y'all.





