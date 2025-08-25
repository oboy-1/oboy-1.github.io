---
layout: post
title:  "So how does PCA actually work?"
categories:
    - 'Learning Interactively'
tags:
    - 'Dimensionality Reduction'
    - 'Principal Component Analysis'

math: true
pin: true
---

In the age of data, making sense of high-dimensional datasets is a common challenge. Principal Component Analysis (PCA) is one of the most powerful tools in the data scientist’s toolkit for reducing dimensionality while preserving the essence of the data. By identifying directions—called principal components—along which the data varies the most, PCA allows us to simplify complex datasets, visualize patterns, and even improve the performance of machine learning models.

> First time hearing about PCA?  Don't worry --- I have an awesome ML Club lecture video on it.  [Access it here]({{ site.baseurl }}/posts/ml-club-video-24-25-dimensionality-reduction/)!
{: .prompt-info }

Before diving into PCA, it’s helpful to have some background knowledge. 

> Here are the key prerequisites:
 - Some idea of what PCA is --- a rough understanding of dimensionality reduction and variance.
 - Linear algebra basics --- matrices, eigenvalues, eigenvectors, and matrix transformations.
 - Multivariable calculus --- knowledge of gradients and, ideally, Lagrange multipliers, though even a basic understanding of partial derivatives is sufficient.
{: .prompt-warning }

With these foundations, you’ll be ready to explore how PCA works under the hood and see why it’s such a versatile technique for analyzing data.

### Variance of Projected Points

Let's focus on an example of converting 2-dimensional data to 1 dimension.  Realize that we essentially have to project the 2D points onto a 1D line.  Try the interactive below to visualize the process.  (Use the orange slider to rotate the line.  Orange points on the graph are the projected points)

<iframe src="{{ site.baseurl }}/assets/files/pca/projection.html" title="Interactive Projection Visualization" style="width: 100%; height: 1000px;" scrolling="no"></iframe>

Now I want you to think, **what line angle preserved the most information**?  If you are having trouble answering that, then here's a clue: when two orange points overlap/cover each other on the projection, we consider that as *information lost* (since two points in different 2D space deceivingly appear as the same on 1D).

> There's actually more to this.  For example, projected points can be completely seperated but equally spaced when the original data is not equally spaced.  This also counts as a loss of information.
{: .prompt-warning }

One could reasonably say that the more the projected points are spread out, the more information is retained.  We essentially want to be able to maximize the **variance** between the projected points. Take a look at the variance indicator in the interactive above and see how it changes as the line rotates. 


![Image]({{ site.baseurl }}/assets/images/2025/08/projection.svg){:.light width="600"}
![Image]({{ site.baseurl }}/assets/images/2025/08/projection_dark.svg){:.dark width="600"}
<figcaption>Vector representation of projection</figcaption>

A projected point is (since we assume $\mathbf{u}$ is a unit vector)

$$
\begin{align*}
y_{\text{proj},i} &= \operatorname{proj}_{\mathbf{u}}(\mathbf{x}_i) \\
                 &= \mathbf{x_i} \cdot \mathbf{u} \\
                 &= \mathbf{x_i}^{T}\mathbf{u}
\end{align*}
$$

Variance (or spread) on these points is defined as the average squared distance from the mean of each point

$$ \operatorname{Var}(y_{proj}) = \frac{1}{n-1} \sum_{i=1}^{n} \left(y_{\text{proj}, i} - \bar{y}\right)^2 $$

$$
y'_{\text{proj},i} = y_{\text{proj},i} - \bar{y}_{\text{proj}}, \quad
\text{where } \bar{y}_{\text{proj}} = \frac{1}{n} \sum_{i=1}^{n} y_{\text{proj},i}
$$

> Why divide by $n-1$ and not $n$?  This is because this is variance of the sample data, not the entire population.  Dividing by $n$ on the sample data always underestimates the variance. Watch [this video from StatQuest](https://www.youtube.com/watch?v=sHRBg6BhKjI) to learn more.
{: .prompt-info }

<iframe src="{{ site.baseurl }}/assets/files/pca/centering.html" title="Interactive Projection Visualization" style="width: 100%; height: 800px;" scrolling="no"></iframe>

Now since the data is now centered, we can say that $\bar{y}_{\text{proj}} = 0$.  This means that 

$$ 
\begin{align*}
\operatorname{Var}(y_{proj}) &= \frac{1}{n-1} \sum_{i=1}^{n} \left(y_{\text{proj}, i} - \bar{y}_{\text{proj}}\right)^2 \\
                             &= \frac{1}{n-1} \sum_{i=1}^{n} \left(y_{\text{proj}, i}\right)^2 \\
                             &= \frac{1}{n-1} \sum_{i=1}^{n} \left(\mathbf{x_i}^{T}\mathbf{u}\right)^2.
\end{align*}
$$

We can now rewrite this as

$$ 
\begin{align*}
\operatorname{Var}(y_{proj}) &= \frac{1}{n-1} \sum_{i=1}^{n} \left(\mathbf{x_i}^{T}\mathbf{u}\right)^2 \\
                             &=  \frac{1}{n-1} \sum_{i=1}^{n} \left(\mathbf{u}^{T}\mathbf{x_i}\right)\left(\mathbf{x_i}^{T}\mathbf{u}\right) \\
                             &=  \frac{1}{n-1} \sum_{i=1}^{n} \mathbf{u}^{T}\left(\mathbf{x_i}\mathbf{x_i}^{T}\right)\mathbf{u} \\
                             &=  \mathbf{u}^{T} \left(\frac{1}{n-1} \sum_{i=1}^{n}\mathbf{x_i}\mathbf{x_i}^{T}\right) \mathbf{u}.
\end{align*}
$$

Now  $\sum_{i=1}^{n}\mathbf{x_i}\mathbf{x_i}^{T}$ is just $XX^{T}$, where $\mathbf{X}$ is the overall matrix $x_i$ is from.  This means that

$$ 
\begin{align*}
\operatorname{Var}(y_{proj}) &= \mathbf{u}^{T} \left(\frac{1}{n-1} \sum_{i=1}^{n}\mathbf{x_i}\mathbf{x_i}^{T}\right) \mathbf{u} \\
                             &= \mathbf{u}^{T} \left(\frac{1}{n-1} \mathbf{X}\mathbf{X}^{T} \right) \mathbf{u}.
\end{align*}
$$

> **Key Point:**<br> $\operatorname{Var}(y_{proj}) = \mathbf{u}^{T} \left(\frac{1}{n-1} \mathbf{X}\mathbf{X}^{T} \right) \mathbf{u}.$
{: .prompt-tip }

### Covariance Matrix

#### Visualizing Covariance

We'll briefly shift to another topic and then bridge these two things together.  I want you to think about the following example: we have a dataset of trees.  We measure two attributes of each tree --- the height and the trunk diameter.

$$
\begin{array}{|l|l|l|}
\hline
\textbf{Tree ID} & \textbf{Trunk Diameter (cm)} & \textbf{Height (m)} \\
\hline
\text{T1} & 5.0 & 3.0 \\
\hline
\text{T2} & 10.0 & 5.0 \\
\hline
\text{T3} & 15.0 & 9.0 \\
\hline
\text{T4} & 20.0 & 11.0 \\
\hline
\text{T5} & 25.0 & 14.0 \\
\hline
\text{T6} & 30.0 & 16.0 \\
\hline
\text{T7} & 35.0 & 18.0 \\
\hline
\end{array}
$$

<iframe src="{{ site.baseurl }}/assets/files/pca/trees.html" title="Trees Plot" style="width: 100%; height: 900px;" scrolling="no"></iframe>

We can plot the data (and center it using the method we previously showed) as seen in the graphs above.  As the graph illustrates, diameter tends to increase as height increases.  But how do we measure that?

We can take each $x_i, y_i$, where $x_i$ is diameter and $y_i$ is height from our **centered** dataset $\mathbf{X}$. We can multiply the coordinates together and sum them for each entry $(x_i, y_i)$, and divide by $n-1$:

> **Key Point:**<br> $\text{Covariance} = \text{Cov}(x,y) = \frac{1}{n-1}\sum_{i=0}^{n} x_i y_i.$
{: .prompt-tip }

What does this value equal?  Lets look at each term individually.  When $x_i$ and $y_i$ are both positive or negative, the product $x_i y_i$ is positive.  Conversely, when $x_i$ and $y_i$ are of opposite signs, the product is negative.

<iframe src="{{ site.baseurl }}/assets/files/pca/trees_red_green.html" title="Trees Plot Red and Green" style="width: 100%; height: 400px;" scrolling="no"></iframe>

This means that, as seen in the graph above, points in the green areas add to the sum, while points in the red areas (if there) would subtract from the sum.

<iframe src="{{ site.baseurl }}/assets/files/pca/try_covariance.html" title="Trees Plot Covariance Yourself" style="width: 100%; height: 670px;" scrolling="no"></iframe>

Click on the graph above to generate points of your own and explore how covariance changes depending on where the points are placed.

> The magnitude of covariance alone conveys very little. It does not indicate how closely the points follow a straight line, nor the steepness of any line of best fit. Moreover, it can be artificially increased or decreased simply by scaling all data points, even though the overall shape and relative distribution of the data remain unchanged.
{: .prompt-danger }

So covariance can be seen as a way of assessing whether two features (in this case tree height and diameter) positively or negatively relate to each other.

> As with many metrics, outliers can through things completely off.  Additionally, note that covariance isn't the same as correlation.
{: .prompt-warning }

#### Variance + Covariance

Now the table of tree data can also be formally written as a matrix:

$$
\mathbf{x} =
\begin{bmatrix}
5.0 & 3 \\
10.0 & 5 \\
15.0 & 9 \\
20.0 & 11 \\
25.0 & 14 \\
30.0 & 16 \\
35.0 & 18
\end{bmatrix}.
$$

Lets multiply this matrix by its transpose

$$
\begin{align*}
    \mathbf{X}\mathbf{X}^T &= 
    \begin{bmatrix}
    5.0 & 3 \\
    10.0 & 5 \\
    15.0 & 9 \\
    20.0 & 11 \\
    25.0 & 14 \\
    30.0 & 16 \\
    35.0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    5.0 & 10.0 & 15.0 & 20.0 & 25.0 & 30.0 & 35.0 \\
    3 & 5 & 9 & 11 & 14 & 16 & 18
    \end{bmatrix} \\
    &= 
    \begin{bmatrix}
    \sum_{i=0}^{n} x_i x_i & \sum_{i=0}^{n} x_i y_i \\
    \sum_{i=0}^{n} y_i x_i & \sum_{i=0}^{n} y_i y_i \\
    \end{bmatrix}.
\end{align*}
$$

And for reasons you'll see soon, I'll multiply that by $\frac{1}{n-1}$:

$$
\begin{align*}
\frac{1}{n-1} \mathbf{X}\mathbf{X}^T &= 
\frac{1}{n-1}
\begin{bmatrix}
    \sum_{i=0}^{n} x_i x_i & \sum_{i=0}^{n} x_i y_i \\
    \sum_{i=0}^{n} y_i x_i & \sum_{i=0}^{n} y_i y_i \\
\end{bmatrix} \\
&= 
\begin{bmatrix}
    \frac{1}{n-1} \sum_{i=0}^{n} x_i x_i & \frac{1}{n-1} \sum_{i=0}^{n} y_i x_i \\
    \frac{1}{n-1} \sum_{i=0}^{n} x_i y_i & \frac{1}{n-1} \sum_{i=0}^{n} y_i y_i \\
\end{bmatrix}.
\end{align*}
$$

Wait! Notice the term $ \frac{1}{n-1} \sum_{i=0}^{n} x_i y_i$.  That is the covariance between height and diameter we calculated before!  Now what is $ \frac{1}{n-1} \sum_{i=0}^{n} x_i x_i$?  Well, 

$$
\begin{align*}
\frac{1}{n-1} \sum_{i=0}^{n} x_i x_i &= \frac{1}{n-1} \sum_{i=0}^{n} \left(x_i\right)^{2} \\
                                     &= \frac{1}{n-1} \sum_{i=0}^{n} \left(x_i - 0\right)^{2}. \\
\end{align*}
$$

And since we centered the data, the mean is $\bar{x_i} = 0$.  This means that 

$$
\begin{align*}
\frac{1}{n-1} \sum_{i=0}^{n} \left(x_i - 0\right)^{2} &= \frac{1}{n-1} \sum_{i=0}^{n} \left(x_i - \bar{x_i}\right)^{2} \\
                                                      &= \text{Var}(x).
\end{align*}
$$

And as we have seen before (see the section "Variance of Projected Points"), the result above is nothing but the variance of $x$!

Putting these things together, we can see that

$$
\begin{align*}
    \frac{1}{n-1} \mathbf{X}\mathbf{X}^T &= 
    \frac{1}{n-1} \begin{bmatrix}
    \sum_{i=0}^{n} x_i x_i & \sum_{i=0}^{n} x_i y_i \\
    \sum_{i=0}^{n} y_i x_i & \sum_{i=0}^{n} y_i y_i \\
    \end{bmatrix}. \\
    &= \frac{1}{n-1} \begin{bmatrix}
    \text{Var}(x) & \text{Cov}(x, y) \\
    \text{Cov}(x, y) & \text{Var}(y) \\
    \end{bmatrix}.
\end{align*}
$$

We define this matrix, which tells us the variance and covariances of different features and combinations of features, respectively, as the **covariance matrix**.  We denote this as

$$

\mathbf{C} = \frac{1}{n-1} \begin{bmatrix}
    \text{Var}(x) & \text{Cov}(x, y) \\
    \text{Cov}(x, y) & \text{Var}(y) \\
    \end{bmatrix}.

$$

### Maximizing Variance

So far we have found the following

$$
\operatorname{Var}(y_{proj}) = \mathbf{u}^{T} \left(\frac{1}{n-1} \mathbf{XX}^{T} \right) \mathbf{u},
$$

$$
\mathbf{C} = \frac{1}{n-1} \mathbf{X}\mathbf{X}^T = \frac{1}{n-1} \begin{bmatrix}
    \text{Var}(x) & \text{Cov}(x, y) \\
    \text{Cov}(x, y) & \text{Var}(y) \\
    \end{bmatrix}.
$$

We can put these two equations together to get 

$$
\operatorname{Var}(y_{proj}) = \mathbf{u}^{T} \mathbf{C} \mathbf{u}.
$$

Now, the entire point of PCA is to maximize this variance.  Also remember that $\mathbf{u}$ is defined to have a magnitude of 1.  To attack this, we can use Calculus --- specifically Lagrange Multipliers.

> If you aren't familiar with Lagrange Multipliers, watch this [awesome video](https://www.youtube.com/watch?v=8mjcnxGMwFo) by Dr. Trefor Bazett that explains it very beautifully.
{: .prompt-info }

Our objective function that we want to maximize is $\operatorname{Var}(y_{proj})$.  We know that

$$

\begin{align*}

\frac{\partial}{\partial \mathbf{u}}\left(\operatorname{Var}(y_{proj})\right)
&= \frac{\partial}{\partial \mathbf{u}}\left(\mathbf{u}^{T} \mathbf{C} \mathbf{u}\right) \\
&= 2\mathbf{C}\mathbf{u}.

\end{align*}

$$

> How did we get that? [This article](https://michael.orlitzky.com/articles/the_derivative_of_a_quadratic_form.xhtml) shows an thorough explanation of how to compute the derivative of a quadratic form, such as $\mathbf{u}^{T} \mathbf{C} \mathbf{u}$
{: .prompt-info }

Our constraint function is that $\lVert \mathbf{u} \rVert = 1$.  We can write this as $\mathbf{u}^T \mathbf{u} = 1$  We know that

$$

\begin{align*}

\frac{\partial}{\partial \mathbf{u}}\left(\mathbf{u}^T \mathbf{u}\right)
&= 2\mathbf{u}.

\end{align*}

$$

Lagrange Multipliers tells us that 

$$

2\mathbf{C}\mathbf{u} = \lambda \cdot 2\mathbf{u}.

$$

for some $\lambda \in \mathbb{R}$.  Simplifying this yields us

> **Key point:**<br> $\mathbf{C}\mathbf{u} = \lambda \mathbf{u}.$
{: .prompt-tip }

But how do we figure out what $\lambda$ is?

### Eigenvectors and Eigenvalues

Let's look again at the equation $\mathbf{C}\mathbf{u} = \lambda \mathbf{u}$.  This is effectively saying that multiplying $\mathbf{u}$ by the matrix $\mathbf{C}$ is equivalent to multiplying $\mathbf{u}$ by a **scalar** quantity $\lambda$.

By definition, we call $\lambda$ as an **eigenvalue**.  From linear algbra, you probably already know that matrices can be seen as ways to transform, or shift, a current coordinate system.  **Eigenvectors** are vectors that, when multiplied by their corresponding matrix, do not change their direction, but just scale by a certain factor (their eigenvalue).

Try out the interactive below to further understand this!

<iframe src="{{ site.baseurl }}/assets/files/pca/eigen.html" title="Interactive Projection Visualization" style="width: 100%; height: 1150px;" scrolling="no"></iframe>

Now you probably can see that the equation $\mathbf{C}\mathbf{u} = \lambda \mathbf{u}$ also is an eigenvector problem: $\mathbf{u}$ is the eigenvector, and $\lambda$ is the eigenvalue.

> Interested to learn more about eigenvectors?  Check out this beautiful [interactive article](https://setosa.io/ev/eigenvectors-and-eigenvalues/) at setosa.io.
{: .prompt-info}


Describing our variance maximization as an eigenvector problem allows us to just solve for the eigenvalues and eigenvectors to get the $\mathbf{u}$ that is the most optimal.  Now in case you are a bit rusty on how to do that --- don't worry, I got you an example! :D
<style>
details {
  background-color: #f0f0f0; /* Light grey background */
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 15px;
}

@media (prefers-color-scheme: dark) {
  details {
    background-color: #333; /* Dark grey background for dark mode */
    color: #eee; /* Adjust text color for better visibility in dark mode */
  }
  details summary {
    color: #eee; /* Adjust summary text color for dark mode */
  }
}

details summary {
  cursor: pointer;
  padding: 10px 0;
}

details > *:not(summary) {
  margin-top: 10px;
}
</style>

{% capture example_content %}
<h5 class="collapsible-heading">Finding Eigenvectors for a 2x2 Matrix by Row Reduction</h5>
<p>Let's find the eigenvectors for the matrix:</p>
$$\mathbf{C} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$$

<p>First, we need to find the <strong>eigenvalues</strong> ($\lambda$) by solving the characteristic equation $\det(\mathbf{C} - \lambda I) = 0$.</p>
$$\mathbf{C} - \lambda I = \begin{bmatrix} 1-\lambda & 2 \\ 2 & 1-\lambda \end{bmatrix}$$
$$(1-\lambda)(1-\lambda) - (2)(2) = 0$$
$$(1-\lambda)^2 - 4 = 0$$
$$(1-\lambda)^2 = 4$$
$$1-\lambda = \pm 2$$
<p>This gives us two eigenvalues:</p>
$$1 - \lambda = 2 \implies \lambda_1 = -1$$
$$1 - \lambda = -2 \implies \lambda_2 = 3$$
    
<br>

<h5 class="collapsible-heading">Step 1: Find the Eigenvector for $\lambda_1 = -1$</h5>

<p>To find the eigenvector $\mathbf{u}_1 = \begin{bmatrix} x \\ y \end{bmatrix}$ corresponding to $\lambda_1 = -1$, we solve the equation $(\mathbf{C} - (-1)I)\mathbf{u}_1 = \mathbf{0}$, which simplifies to $(\mathbf{C}+I)\mathbf{u}_1 = \mathbf{0}$.</p>

$$\mathbf{C} + I = \begin{bmatrix} 1+1 & 2 \\ 2 & 1+1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$$

<p>We use Gaussian elimination on the augmented matrix:</p>
$$\begin{bmatrix} 2 & 2 & | & 0 \\ 2 & 2 & | & 0 \end{bmatrix}$$
<p>Row reduction: $R_2 \to -R_1 + R_2$:</p>
$$\begin{bmatrix} 2 & 2 & | & 0 \\ 0 & 0 & | & 0 \end{bmatrix}$$

<p>This gives the equation: $2x + 2y = 0 \implies x = -y$. The variable $y$ is free. If we let $y=1$, then $x=-1$.
Therefore, a basis for the eigenspace of $\lambda_1 = -1$ is the vector:</p>

$$\mathbf{u}_1 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

<br>

<h5 class="collapsible-heading">Step 2: Find the Eigenvector for $\lambda_2 = 3$</h5>

<p>For the second eigenvalue $\lambda_2 = 3$, we solve the equation $(\mathbf{C} - 3I)\mathbf{u}_2 = \mathbf{0}$.</p>

$$\mathbf{C} - 3I = \begin{bmatrix} 1-3 & 2 \\ 2 & 1-3 \end{bmatrix} = \begin{bmatrix} -2 & 2 \\ 2 & -2 \end{bmatrix}$$

<p>We use Gaussian elimination on the augmented matrix:</p>
$$\begin{bmatrix} -2 & 2 & | & 0 \\ 2 & -2 & | & 0 \end{bmatrix}$$
<p>Row reduction: $R_2 \to R_1 + R_2$:</p>
$$\begin{bmatrix} -2 & 2 & | & 0 \\ 0 & 0 & | & 0 \end{bmatrix}$$

<p>This gives the equation: $-2x + 2y = 0 \implies x = y$. The variable $y$ is free. If we let $y=1$, then $x=1$.
Therefore, a basis for the eigenspace of $\lambda_2 = 3$ is the vector:</p>

$$\mathbf{u}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

{% endcapture %}

{% include collapsible.html summary="Click here for a an example of solving for eigenvectors" content=example_content %}

Well... there is one caveat.  We most often get **two** eigenvectors as solutions for $\mathbf{u}$ --- so which one it it?  When we were maximizing variance, we forgot to account that the derivative being 0 can also mean that the value at that point is a minimum.  So one eigenvector gives us the maximum variance, and the other the miniumum.  With multiple eigenvalue components for dimensions greater than 2, the magnitude of the eignenvalue corresponds to how much variance projecting along that vector line yields.

But why?  Remember that 

$$

\operatorname{Var}(y_{proj}) = \mathbf{u}^{T} \mathbf{C} \mathbf{u}.

$$

And since $\mathbf{C}\mathbf{u} = \lambda \mathbf{u},$

$$

\operatorname{Var}(y_{proj}) = \mathbf{u}^{T} \lambda \mathbf{u}.

$$

This indicates that the greater the eigenvalue $\lambda$, the greater the variance.  Now try the projection interactive again, as shown below, and align the line towards either of the principal components shown in red.  Notice how one maximizes and the other minimizes the variance.

<iframe src="{{ site.baseurl }}/assets/files/pca/projection_pca_always_show.html" title="Interactive Projection Visualization" style="width: 100%; height: 1000px;" scrolling="no"></iframe>

### In Practice

That's it for the theory.  But before we call it off, let me summarize the process of PCA --- and showcase how this looks like beyond 2-dimensions.

We started with a dataset.  Let's say this dataset has $N$ entries, each with $M$ features.  We represent this as 

$$
\mathbf{X} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1M} \\
x_{21} & x_{22} & \cdots & x_{2M} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} & x_{N2} & \cdots & x_{NM}
\end{bmatrix}
$$

We then calculate the corresponding covariance matrix (note that we are both centering the data by subtracting the mean and multiplying the matrix by its transpose here):

$$
\begin{align*}
\frac{1}{N-1}(\mathbf{X} - \bar{x})^T(\mathbf{X} - \bar{x}) &= \begin{bmatrix}
\text{cov}(x_1, x_1) & \text{cov}(x_1, x_2) & \cdots & \text{cov}(x_1, x_N) \\
\text{cov}(x_2, x_1) & \text{cov}(x_2, x_2) & \cdots & \text{cov}(x_2, x_N) \\
\vdots & \vdots & \ddots & \vdots \\
\text{cov}(x_N, x_1) & \text{cov}(x_N, x_2) & \cdots & \text{cov}(x_N, x_N)
\end{bmatrix} \\
&= \mathbf{C}.
\end{align*}
$$

Now we have to find the eigenvalues and eigenvectors of $\mathbf{C}$.  That is, we have to solve the equation,

$$
\mathbf{C}\mathbf{u} = \lambda \mathbf{u}.
$$

This is can be solved through setting $\det(A - \lambda I) = 0$ and solving for $\lambda$ for eigenvalues.  Then we just, for each $\lambda$ solution, solve for $\mathbf{u}$ in the equation $\left(\mathbf{C} - \lambda \mathbf{I}\right)\mathbf{u} = 0$. (Check out the grey box dropdown in the section above for an example on this)

The options for the vector $u$ with the highest eigenvalues are the ones you should project your data on to.  Depending on the number of dimensions you want to decrease your data to, you will choose that nymber of eigenvectors to use for projection.  These eigenvectors you just chose for projection are called your **principal components**.

> **Key Point:** <br> The eigenvectors of the covariance matrix are the principal components of the data.
{: .prompt-tip }

Checkout the 3D to 2D example below!

<iframe src="{{ site.baseurl }}/assets/files/pca/projection_3d.html" title="Interactive 3D Projection Visualization" style="width: 100%; height: 900px;" scrolling="no"></iframe>

And that's really it!  Hope you got a deeper understanding of PCA!

Thanks for reading!
