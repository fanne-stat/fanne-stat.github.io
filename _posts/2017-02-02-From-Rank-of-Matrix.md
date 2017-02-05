---
layout: post
title: "From the Rank of Matrix"
categories: journal
tags: [math]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---


Recently the Linear Model course that involves a lot about linear algebra gives me an opportunity to review some linear algebra materials I studied before. Although what the course need is all about matrices, but I always hold this opinion that to have a better understanding of the matrix things (the general linear properties), linear transformation and linear space should be definitely the best way. To understand it geometrically helps us to have an intuition about some results about matrix.

## Matrix from the view of linear transformation

So what is a matrix? If it is an $n\times m$ matrix, then we can apply it to a vector of length $m$ and get a vector of length $n$. In other words, it is a transformation from $\mathbb{R}^m$ to $\mathbb{R}^n$. However, it can be more than this. 

Let's be more abstract. Given two linear space $V$ and $W$ and a linear transformation $\mathcal{A}: V \to W$, how can we make a profile of the linear transformation?

A natural idea is to find the image of all the elements of the original linear space $V$ in the destination linear space $W$. With this we know the mapping completely. But this method can actually work in any spaces and any transformations, and is too general for the special case of linear space and linear transformation. In fact, in this case what we need is only the image of all the basis of the original linear space $V$.

The reason is we can always write any elements in $V$ as a linear combination the basis of $V$. And this combination is unique. The uniqueness can be obtained like this:

For an arbitrary element $\boldsymbol{x}$ in $V$ with basis $\\{\boldsymbol{\epsilon}_1, \boldsymbol{\epsilon}_2, \cdots, \boldsymbol{\epsilon}_n\\}$, if we have two linear combinations of basis that give

$$
\begin{align*}
&\boldsymbol{x} = a_1 \boldsymbol{\epsilon}_1 + a_2 \boldsymbol{\epsilon}_2 + \cdots + a_n \boldsymbol{\epsilon}_n\\
&\boldsymbol{x} = b_1 \boldsymbol{\epsilon}_1 + b_2 \boldsymbol{\epsilon}_2 + \cdots + b_n \boldsymbol{\epsilon}_n\\
\end{align*}$$

Then we have 

$$
(a_1 -b_1) \boldsymbol{\epsilon}_1 + (a_2 - b_2) \boldsymbol{\epsilon}_2 + \cdots + (a_n - b_n) \boldsymbol{\epsilon}_n = \boldsymbol{0}
$$

As basis are linearly independent, then we have $a_i = b_i$ for all $i = 1, \cdots, n$. Thus it is unique.

So for an element in $V$, we can decompose it as $\boldsymbol{x} = a_1 \boldsymbol{\epsilon}_1 + a_2 \boldsymbol{\epsilon}_2 + \cdots + a_n \boldsymbol{\epsilon}_n$. Then the image of $\boldsymbol{x}$ would be 

$$\mathcal{A}\boldsymbol{x} = \mathcal{A}(a_1 \boldsymbol{\epsilon}_1 + a_2 \boldsymbol{\epsilon}_2 + \cdots + a_n \boldsymbol{\epsilon}_n) = a_1 \mathcal{A} \boldsymbol{\epsilon}_1 + a_2 \mathcal{A} \boldsymbol{\epsilon}_2 + \cdots + a_n \mathcal{A} \boldsymbol{\epsilon}_n$$

Thus when we know the image of basis, we know the image of all elements.

As $\mathcal{A}\boldsymbol{\epsilon}_i \in W$, then we can represent it uniquely as 

$$\mathcal{A} \boldsymbol{\epsilon}_i = a_{1i} \boldsymbol{\eta}_1 + a_{2i} \boldsymbol{\eta}_2 + \cdots + a_{mi} \boldsymbol{\eta}_m$$

where $\\{\boldsymbol{\eta}_1,\boldsymbol{\eta}_2,,\cdots, \boldsymbol{\eta}_m \\}$ is basis of $W$.

To be more clear we have 

$$
\begin{align*}
\mathcal{A} \begin{bmatrix}
\boldsymbol{\epsilon}_1 & \boldsymbol{\epsilon}_2 & \cdots & \boldsymbol{\epsilon}_n
\end{bmatrix} &= 
\begin{bmatrix}
\boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_m
\end{bmatrix}
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn} 
\end{bmatrix} 
\\&= \begin{bmatrix}
\boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_m
\end{bmatrix} \boldsymbol{A}
\end{align*}$$

Then we can see $\boldsymbol{A}$ is the representation of $\mathcal{A}$ under the given basis of two linear space. From this point of view, for two invertible matrices $\boldsymbol{P}$ and $\boldsymbol{Q}$, we can take $\boldsymbol{P}^{-1} \boldsymbol{A} \boldsymbol{Q}$ and $\boldsymbol{A}$ as the same. In this case, the new basis used are $\begin{bmatrix}
\boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_m
\end{bmatrix} \boldsymbol{P}$ and $\begin{bmatrix}
\boldsymbol{\epsilon}_1 & \boldsymbol{\epsilon}_2 & \cdots & \boldsymbol{\epsilon}_n
\end{bmatrix} \boldsymbol{Q}$. Thus the representation under the new basis is

$$
\mathcal{A} \begin{bmatrix}
\boldsymbol{\epsilon}_1 & \boldsymbol{\epsilon}_2 & \cdots & \boldsymbol{\epsilon}_n
\end{bmatrix} \boldsymbol{Q} = \begin{bmatrix}
\boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_m
\end{bmatrix} \boldsymbol{P} (\boldsymbol{P}^{-1} \boldsymbol{A} \boldsymbol{Q})$$

## Rank 

We have known that matrix is a representation of linear transformation, so we can define the rank of a linear transformation first.

**Definition** - Rank of linear transformation $\mathcal{A}$ is the dimension of the image space of $\mathcal{A}$, denoted as $\mathcal{A} V$, where $V$ is the original space.

Then we can define a rank of matrix $\boldsymbol{A}$ as the rank of the linear transformation $\mathcal{A}$. From this definition we can also know $rank(\boldsymbol{A}) = rank(\boldsymbol{P}^{-1} \boldsymbol{A} \boldsymbol{Q})$. They are just two different representations of the same linear transformation.

For a given basis of the original space $V$, say $\\{\boldsymbol{\epsilon}_1, \boldsymbol{\epsilon}_2, \cdots, \boldsymbol{\epsilon}_n\\}$, the image space can also be written as $span\\{\mathcal{A}\boldsymbol{\epsilon}_1,\mathcal{A}\boldsymbol{\epsilon}_2, \cdots, \mathcal{A}\boldsymbol{\epsilon}_n \\}$. 

From the previous discussion we know $\mathcal{A}\boldsymbol{\epsilon}_i$ corresponds to the $i$-th column of $\boldsymbol{A}$.

$$\mathcal{A} \boldsymbol{\epsilon}_i = \begin{bmatrix}
\boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_m 
\end{bmatrix} \boldsymbol{a}_i $$

$\boldsymbol{a}_i$ is the $i$-th column of $\boldsymbol{A}$. There is a one-to-one mapping between them. Then it is easy to prove that the linear structure of $\\{\mathcal{A}\boldsymbol{\epsilon}_1,\mathcal{A}\boldsymbol{\epsilon}_2, \cdots, \mathcal{A}\boldsymbol{\epsilon}_n \\}$ is the same as that of $\\{\boldsymbol{a}_1, \boldsymbol{a}_2, \cdots, \boldsymbol{a}_n\\}$. Hence the image space of $\mathcal{A}$ can also be seen as $span\\{\boldsymbol{a}_1, \boldsymbol{a}_2, \cdots, \boldsymbol{a}_n\\} = \mathcal{C}(\boldsymbol{A})$, which is also know as column space of $\boldsymbol{A}$. When we are talking about a linear transformation from $\mathbb{R}^n$ to $\mathbb{R}^m$ and take the standard basis of each space, then image space is exactly the column space. So we can also call $\mathcal{C}(\boldsymbol{A})$ the image space of $\boldsymbol{A}$.

### Null space of $\mathcal{A}$

We define the null space of $\mathcal{A}$ as $\mathcal{A}^{-1}(\boldsymbol{0}) = \\{\boldsymbol{v}: \mathcal{A}\boldsymbol{v} = \boldsymbol{0}\\}$. Then we have a theorem about the image space and null space:

**Theorem** - The original image of a basis of the image space of $\mathcal{A}$ and the basis of null space can form a basis of the original space. 

Assume the basis of the image space is 
$\\{\boldsymbol{\eta}_{1}, \boldsymbol{\eta}_2, \cdots, \boldsymbol{\eta}_r\\}$, 
and their corresponding original images are $\\{\boldsymbol{\epsilon}_1, \cdots, \boldsymbol{\epsilon}_r\\}$, which means
$\mathcal{A}\boldsymbol{\epsilon}_i = \boldsymbol{\eta}_i$.   
It is not hard to prove that $\\{\boldsymbol{\epsilon}_1, \cdots, \boldsymbol{\epsilon}_r\\}$ are linearly independent. 

Then assume the basis of $\mathcal{A}^{-1}(\boldsymbol{0})$ is $\\{\boldsymbol{\epsilon}_{r}, \cdots, \boldsymbol{\epsilon}_m\\}$.
 We first show $\\{\boldsymbol{\epsilon}_1, \cdots , \boldsymbol{\epsilon}_r , \boldsymbol{\epsilon}\_{r+1}, \cdots, \boldsymbol{\epsilon}\_m\\}$ are linearly independent. 

 Let $l_1 \boldsymbol{\epsilon}_1 + l_2 \boldsymbol{\epsilon}_2 + \cdots + l_m \boldsymbol{\epsilon}_m = \boldsymbol{0}$. 
 Applying $\mathcal{A}$ to both sides, 
 we have 

 $$l_1 \mathcal{A} \boldsymbol{\epsilon}_1 + \cdots + l_r \mathcal{A} \boldsymbol{\epsilon}_r + l\_{r+1} \mathcal{A} \boldsymbol{\epsilon}\_{r+1}+ \cdots + l_m \mathcal{A} \boldsymbol{\epsilon}_m = l_1 \boldsymbol{\eta}_1 + \cdots + l_r \boldsymbol{\eta}_r = \boldsymbol{0}$$
  
 As $\\{\boldsymbol{\eta}_1, \cdots , \boldsymbol{\eta}_r\\}$ are linearly independent, then we have $l_1 = \cdots = l_r = 0$. Thus $l\_{r+1} \boldsymbol{\epsilon}\_{r+1} + \cdots + l_m \boldsymbol{\epsilon}_m = \boldsymbol{0}$. Again, $\\{\boldsymbol{\epsilon}\_{r+1}, \cdots, \boldsymbol{\epsilon}_m\\}$ are linearly independent, then we have $l\_{r+1} = \cdots = l_m = 0$. Hence from $l_1 \boldsymbol{\epsilon}_1 + \cdots + l_m \boldsymbol{\epsilon}_m = \boldsymbol{0}$ we get $l_1 = \cdots = l_m = 0$. Thus $\\{\boldsymbol{\epsilon}_1 ,\cdots , \boldsymbol{\epsilon}_m\\}$ are linearly independent.

 We then prove any $\boldsymbol{v} \in V$ can be written as linear combination of $\\{\boldsymbol{\epsilon}_1, \cdots, \boldsymbol{\epsilon}_m\\}$. 

 First $\mathcal{A} \boldsymbol{v} \in \mathcal{A}V$, then we have $\mathcal{A} \boldsymbol{v} = h_1 \boldsymbol{\eta}_1 + \cdots + h_r \boldsymbol{\eta}_r  = h_1 \mathcal{A} \boldsymbol{\epsilon}_1 + \cdots + \boldsymbol{\epsilon}_r  = \mathcal{A}(h_1 \boldsymbol{\epsilon}_1 + \cdots + h_r \boldsymbol{\epsilon}_r) $
 $\Rightarrow \mathcal{A}(\boldsymbol{v} - (h_1 \boldsymbol{\epsilon}_1 + \cdots + h_r \boldsymbol{\epsilon}_r)) = \boldsymbol{0}$. Thus $\boldsymbol{v} - (h_1 \boldsymbol{\epsilon}_1 + \cdots + \boldsymbol{\epsilon}_r) \in \mathcal{A}^{-1}(\boldsymbol{0})$. Then we have $\boldsymbol{v} - (h_1 \boldsymbol{\epsilon}_1 + \cdots + h_r \boldsymbol{\epsilon}_r) = h\_{r+1} \boldsymbol{\epsilon}\_{r+1} + \cdots + h_m \boldsymbol{\epsilon}_m$. Thus any $\boldsymbol{v} \in V$ can be represented as linear combination of the linearly independent set $\\{\boldsymbol{\epsilon}_1 , \cdots , \boldsymbol{\epsilon}_m\\}$, thus it is a basis of $V$. And from this we can also know $m = \dim{V}$.

 One simple conclusion we can draw from this theorem is 

 $$\dim{\mathcal{A}V} + \dim{\mathcal{A}^{-1}(\boldsymbol{0})} = \dim{V}$$

  The dimension of image space plus the dimension of null space is equal to the dimension of the original space.

  For a $m\times n$ matrix, the image space is the column space, then

  $$ \dim\mathcal{C}(\boldsymbol{A}) + \dim \boldsymbol{A}^{-1}(\boldsymbol{0}) = rank(\boldsymbol{A}) + \dim \boldsymbol{A}^{-1}(\boldsymbol{0}) = n $$  

  We can actually view the null space of matrix in another way. Let $\boldsymbol{a}\_{(i)}$ be the $i$-th row of matrix $\boldsymbol{A}$. Then the row space of $\boldsymbol{A}$ is $\mathcal{R}(\boldsymbol{A}) = span\\{\boldsymbol{a}\_{(1)}, \cdots, \boldsymbol{a}\_{(m)}\\}$. So now 

  $$\boldsymbol{A} \boldsymbol{x} = \boldsymbol{0} \iff \begin{bmatrix}
  \boldsymbol{a}_{(1)}^T\\
  \vdots\\
  \boldsymbol{a}_{(m)}^T
  \end{bmatrix}\boldsymbol{x} = \begin{bmatrix}
  \boldsymbol{a}_{(1)}^T \boldsymbol{x}\\
  \vdots\\
  \boldsymbol{a}_{(m)}^T \boldsymbol{x}
  \end{bmatrix} = \boldsymbol{0}$$ 

  We can see the null space of $\boldsymbol{A}$ is actually the orthogonal compliment of row space $\mathcal{R}(\boldsymbol{A})$ in $\mathbb{R}^n$. Then we have 

  $$\dim \mathcal{R}(\boldsymbol{A}) + \dim \mathcal{A}^{-1}(\boldsymbol{0}) = n$$

  By comparing it with the previous equation, we have 

  $$\dim \mathcal{C}(\boldsymbol{A}) = \dim\mathcal{R}(\boldsymbol{A}) = rank(\boldsymbol{A}).$$

## Understand some results in an easy way

Denote $\boldsymbol{P}\_{\boldsymbol{X}}$ be the projection matrix that project vectors to the column space of $\boldsymbol{X}$.

1. $rank(\boldsymbol{P}\_{\boldsymbol{X}}) = rank(\boldsymbol{X})$

    Since $\boldsymbol{P}\_{\boldsymbol{X}}$ and $\boldsymbol{X}$ have the same image space (column space), this result is obvious. 

2. $rank(\boldsymbol{A} \boldsymbol{B}) \leq \min\\{rank(\boldsymbol{A}), rank(\boldsymbol{B})\\}$

    Let $\mathcal{A}$ and $\mathcal{B}$ be the corresponding linear transformation, then $rank(\boldsymbol{A}\boldsymbol{B}) = rank(\mathcal{A}\mathcal{B})$. Let $\mathcal{B}V$ be the image space of $\mathcal{B}$, then $\mathcal{A}\mathcal{B}$ is actually $(\mathcal{A}\|\mathcal{B}V) \mathcal{B}$. Here $\mathcal{A}\|\mathcal{B}V$ is the linear transformation $\mathcal{A}$ restricted on the image space of $\mathcal{B}$. And then the image space of $\mathcal{A}\mathcal{B}$ is just the image space of $\mathcal{A}\|\mathcal{B}V$. It is easy to understand the dimension of the image space of $\mathcal{A}\|\mathcal{B}V$ cannot be larger than the dimension of $\mathcal{B}V$. Because here $\mathcal{B}V$ is the original space, and that statement is true because $\mathcal{A}$ is a linear tranformation. Also, the image space of $\mathcal{A}\|\mathcal{B}V$ cannot be larger than the image space of $\mathcal{A}$. It is because the former one is the transformation of the latter one restricted in a subspace.

    Thus we can see $rank(\mathcal{A}\mathcal{B}) \leq rank(\mathcal{A})$ and $rank(\mathcal{A}\mathcal{B}) \leq rank(\mathcal{B})$. Writing it in the view of corresponding matrix would be $rank(\boldsymbol{A}\boldsymbol{B}) \leq \min\\{rank(\boldsymbol{A}), rank(\boldsymbol{B})\\}$
 
3. $rank(\boldsymbol{A}\boldsymbol{P}\_{\boldsymbol{X}}) = rank(\boldsymbol{A}\boldsymbol{X})$

    As $\boldsymbol{P}\_{\boldsymbol{X}} V = \boldsymbol{X}V = \mathcal{C}(\boldsymbol{X})$, then $rank(\boldsymbol{A}\boldsymbol{P}\_{\boldsymbol{X}}) = rank(\boldsymbol{A}\boldsymbol{X}) = rank(\mathcal{A}\|\mathcal{C}(\boldsymbol{X}))$.

4. $rank(\boldsymbol{X}) = rank(\boldsymbol{X}^T \boldsymbol{X})$

	In fact, $\boldsymbol{X}$ and $\boldsymbol{X}^T \boldsymbol{X}$ have the same null space.

	For any $\boldsymbol{y}$ in null space of $\boldsymbol{X}$, we have $\boldsymbol{X}\boldsymbol{y}=\boldsymbol{0}$, then we certainly have $\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{y} = \boldsymbol{0}$.

	For any $\boldsymbol{y}$ such that $\boldsymbol{X}^T \boldsymbol{X} \boldsymbol{y} = \boldsymbol{0}$, we can multiply the left side with $\boldsymbol{y}^T$, then $\boldsymbol{y}^T \boldsymbol{X}^T \boldsymbol{X}\boldsymbol{y} = (\boldsymbol{X}\boldsymbol{y})^T \boldsymbol{Xy} = \boldsymbol{0}$. Hence $\boldsymbol{Xy} = \boldsymbol{0}$.

	Another way to prove is:

	$\boldsymbol{X}^T \boldsymbol{Xy} = \boldsymbol{0} \Rightarrow \boldsymbol{Xy}$ is in the null space of $\boldsymbol{X}^T$. Then $\boldsymbol{Xy}$ should be in the orthogonal compliment of row space of $\boldsymbol{X}^T$, which is also the orthogonal compliment of column space of $\boldsymbol{X}$. But $\boldsymbol{Xy}$ is also in the column space of $\boldsymbol{X}$. So it can only be in the intersection of them, which can only be $\boldsymbol{0}$. Thus $\boldsymbol{Xy} = \boldsymbol{0}$.

	Since we know $\boldsymbol{X}$ and $\boldsymbol{X}^T \boldsymbol{X}$ have the same null space, then 

	$$rank(\boldsymbol{X}) = rank(\boldsymbol{X}^T \boldsymbol{X}) = n - \dim(\textrm{null space})$$

## Idempotent matrix and projection matrix

For an idempotent matrix $\boldsymbol{A}$, let $\mathcal{A}$ be its corresponding linear transformation. Then if we take a basis of $\mathcal{A}V$,
 $\\{\boldsymbol{\eta}\_i\\}\_{i=1}^r$. Assume the original image of $\boldsymbol{\epsilon}_i$, then $\mathcal{A}(\mathcal{A} \boldsymbol{\epsilon}_i) = \mathcal{A}\boldsymbol{\epsilon}_i \Rightarrow \mathcal{A} \boldsymbol{\eta}_i = \boldsymbol{\eta}_i$. Thus their original images are just $\\{\boldsymbol{\eta}_i\\}\_{i=1}^r$. Thus $\\{\boldsymbol{\eta}_1, \cdots, \boldsymbol{\eta}_r\\}$ and the basis of null space $\\{\boldsymbol{\eta}\_{r+1},\cdots, \boldsymbol{\eta}_n\\} $can form a basis of $V$. Under this basis

 $$\mathcal{A}\begin{bmatrix}
 \boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_r & \boldsymbol{\eta}_{r+1} & \cdots & \boldsymbol{\eta}_n
 \end{bmatrix} = \begin{bmatrix} \boldsymbol{\eta}_1 & \boldsymbol{\eta}_2 & \cdots & \boldsymbol{\eta}_r & \boldsymbol{\eta}_{r+1} & \cdots & \boldsymbol{\eta}_n
 \end{bmatrix} 
  \begin{bmatrix}
 1 & & & & & &\\
   & 1& & & & &\\
   &  &\ddots & & & &\\
   &  & & 1 & & &\\
   & & & & 0 & &\\
   & & & & & \ddots & \\
   & & & & & & 0
 \end{bmatrix}
$$

From this representation we can see idempotent matrix is to some extent a "projection matrix". It project vector to $span\\{\boldsymbol{\eta}_1, \cdots, \boldsymbol{\eta}_r\\}$. However such project may not be a orthogonal projection. As we can see the null space and $span\\{\boldsymbol{\eta}_1, \cdots, \boldsymbol{\eta}_r\\}$ (image space, column space) might not be orthogonal.

To make it an orthogonal projection, we just need to add one condition: $\boldsymbol{A}$ is symmetric. Then the column space is also the row space. We already know row space and null space are mutually orthogonal compliment, then in this case the column space (image space) and null space are orthogonal.

When the projection is orthogonal projection, then $\boldsymbol{A}$ becomes what we usually call projection matrix. Hence we know

$\boldsymbol{P}$ is a projection matrix $\iff$ $\boldsymbol{P}^2 = \boldsymbol{P}$ and $\boldsymbol{P}^T = \boldsymbol{P}$.  
 


