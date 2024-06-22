---
id: 152
title: 'Coding with Karthik: Visualizing the Perceptron Learning Algorithm'
date: '2024-01-05T03:13:07+00:00'

layout: post
guid: 'http://karthikvedula.com/?p=152'
permalink: /2024/01/05/visualizing-the-perceptron-learning-algorithm/
footnotes:
    - ''
categories:
    - 'Coding with Karthik'
    - 'Machine Learning'
    - Uncategorized
tags:
    - 'Machine Learning'
    - 'Perceptron Learning'
    - Programming
---

The Perceptron Learning Algorithm is a classification algorithm that generates a line such that it fully separates the data into two classes. This is done by starting off with a randomly generated line, and then the lines slope (weight) and its intercept (bias) are updated according the misclassified points. This happens until the line classifies **all** points correctly (so if the data isn’t linearly separable, this algorithm will run forever…).

Walk through the code and see a cool animation on the bottom!

### 1. Prepare dataset

Create linearly seperable data (initially vertical/horizontal) and then rotate it


```python
from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('default')
```


```python
separable = False
while not separable:
    samples = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)
    red = samples[0][samples[1] == 0]
    blue = samples[0][samples[1] == 1]
    separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
plt.plot(red[:, 0], red[:, 1], 'r.')
plt.plot(blue[:, 0], blue[:, 1], 'b.')
plt.show()
```


    
![png]({{ site.baseurl }}/assets/images/blog_perceptronVis_files/blog_perceptronVis_2_0.png)
    



```python
import numpy as np

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

origin=(np.mean(samples[0], axis=0))
new_samples_coords = rotate(samples[0], origin=origin, degrees=np.random.randint(30, 80))
samples  = (new_samples_coords, samples[1])
```


```python
red = samples[0][samples[1] == 0]
blue = samples[0][samples[1] == 1]

plt.plot(red[:, 0], red[:, 1], 'r.')
plt.plot(blue[:, 0], blue[:, 1], 'b.')
plt.show()
```


    
![png]({{ site.baseurl }}/assets/images/blog_perceptronVis_files/blog_perceptronVis_4_0.png)
    


Create test dataset


```python
red_test = red[:20]
blue_test = blue[:20]
samples_test = (samples[0][:20], samples[1][:20]) 

red = red[20:]
blue = blue[20:]
samples = (samples[0][20:], samples[1][20:]) 
```

### 2. Train perceptron on the data

The function below returns samples that were misclassified by the line $y = b + \vec{w}_1 \cdot \vec{x}_2 + \vec{w}_2 \cdot \vec{x}_2$


```python
def get_misclassified(samples, w1, w2, b): 
    res = []
    
    for i in range(len(samples[0])):
        x1, x2 = samples[0][i]
        pred = b + w1*x1 + w2*x2 
        Y = samples[1][i]
        
        if (pred > 0) != Y:
            res.append((x1, x2, Y))
            
    return res
```

The learning rate, "LR", scales how much we correct our perceptron for each misclassified point.  High LR means we overcorrect, meaning the line will bounce around the data more.  Low LR means that the we undercorrect, meaning the line will not move much and hence will take a longer time to train.


```python
LR = 1
```

The code below is what trains our perceptron.  Look out for the comments to understand how it works.


```python
!mkdir animation # make the folder that will store our results
```

    mkdir: animation: File exists



```python
# Start off with random weights
B = np.random.randint(-10,10)
W1 = np.random.randint(-10,10)
W2 = np.random.randint(-10,10)

count = 0 # used for ordering the intermediate steps that will be also outputed

misclassified = get_misclassified(samples, W1, W2, B)
while len(misclassified) != 0:
    x = np.linspace(-2, 2) # used as the x values to plot our line
    
    for sample in misclassified:
        plt.close() # delete the old line

        # get the sample values
        x1 = sample[0]
        x2 = sample[1]
        y = sample[2]

        if y == 0: y = -1 # change y so that when we update weights it is in the scale of -1 or 1 rather than 0 or 1

        # update weights
        B = B + y * 1 * LR
        W1 = W1 + y * x1 * LR
        W2 = W2 + y * x2 * LR

        ax = plt.gca()
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

        # plot and save the figure
        plt.title(f'frame number {count}')
        plt.plot(x, -W1/W2 * x - B/W2) # plot line
        plt.plot(red[:, 0], red[:, 1], 'r.') # plot red points
        plt.plot(blue[:, 0], blue[:, 1], 'b.') # plot blue points

        plt.plot(x1, x2, 'o') # point being used to correct the line
        
        plt.savefig(f'./animation/{count}') # save the figure, naming it as its index "count"
        
        count += 1

    
    misclassified = get_misclassified(samples, W1, W2, B) # calculate the new misclassified points

```

![png]({{ site.baseurl }}/assets/images/blog_perceptronVis_files/blog_perceptronVis_14_0.png)
    

The code below converts the images into a gif

```python
import glob
import contextlib
from PIL import Image

import re

def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def human_sort(l):
    """
    Sort a list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

# filepaths
fp_in = "./animation/*.png"
fp_out = "./animation/out.gif"

files = human_sort(glob.glob(fp_in))

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in files)

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=400, loop=0)
```

**Results are in the 'animations' folder in the same directory as this notebook.  'out.gif' is the animation of the perceptron learning**

### 3. Test the model


```python
plt.plot(x, -W1/W2 * x - B/W2) # plot line

plt.plot(red[:, 0], red[:, 1], 'r.') # red train data
plt.plot(blue[:, 0], blue[:, 1], 'b.') # blue train data

plt.plot(red_test[:, 0], red_test[:, 1], 'g.') # red TEST points
plt.plot(blue_test[:, 0], blue_test[:, 1], 'y.') # blue TEST points
```




    [<matplotlib.lines.Line2D at 0x12c5c4a00>]




    
![png]({{ site.baseurl }}/assets/images/blog_perceptronVis_files/blog_perceptronVis_19_1.png)
    
Below is the gif that was created from the above code. I can look at this for hours 😀

![Image]({{ site.baseurl }}/assets/images/2024/01/out.gif)
