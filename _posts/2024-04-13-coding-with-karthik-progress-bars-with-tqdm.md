---
id: 342
title: 'Coding with Karthik: Progress bars with TQDM'
date: '2024-04-13T14:47:55+00:00'

layout: post
guid: 'http://karthikvedula.com/?p=342'
permalink: /2024/04/13/coding-with-karthik-progress-bars-with-tqdm/
footnotes:
    - ''
categories:
    - 'Coding with Karthik'
    - 'Machine Learning'
tags:
    - Programming
    - TQDM
---

“Are we there yet?”

We all probably must have asked this question *multiple times* during long road trips when we were young. Why was this always a burning question? Chances are because we didn’t know how much time was left!

In the digital world, while there aren’t any long drives, there are long uploads/downloads, long rendering processes, and long software updates just to say the least. To make all of these tasks not seem like an eternity, programmers included progress bars to quench the user’s curiosity of how much more time they need to wait.

**`tqdm` is a tool that you can use to include these progress bars in Python!**

---

### 1. Intro

`tqdm` can be wrapped around an iterator to show the progress of iterating through the iterator.
```python
from tqdm import tqdm

for item in tqdm(range(1000)):
    # Do something with item
```

This yields:

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-12-at-7.49.35-PM-1024x105.png)

### 2. Description

You can also customize the progress bar to have a description, with the `desc` parameter.

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-12-at-8.30.08-PM-1024x67.png)

### 3. Nested `for` Loops

If you just use `tqdm` when using multiple `for` loops, you get the following result:

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-13-at-9.05.00-AM-1024x156.png)

This leaves a lot of previously finished inner-loop progress bars on the screen, making it a bit messy. To counteract this, just put `leave=False` as one of the parameters in the inner loop. This will make the inner loop bar disappear once it is done.

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-13-at-9.40.30-AM-1024x156.png)

### 4. More control of the progress bar

You can also control the progress bar even more with the **`with`** keyword in python. Here is an example:
```python
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to train mode
    running_loss = 0.0

    # Iterate over the training dataset with tqdm
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for images, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'Loss': running_loss / ((pbar.n - 1) * train_loader.batch_size + images.size(0))})
```

In the example above, we are training an ML model in PyTorch. There are two `for` loops that are nested: one for the epochs and another for the different samples in the dataset. In the inner `for` loop, there is the `desc`** parameter that states the epoch number. Within this `for` loop, notice the `pbar.update(1)` and `pbar.set_postfix("...")` methods used. The update method advances the progress bar progress by the argument provided, and the postfix method adds information on the right side of the progress bar (in this case the loss).

This is the output of that code:

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-13-at-10.07.43-AM-1024x156.png)

### 5. `tqdm` in Jupyter Notebooks

You can make `tqdm` look much better in [Jupyter Notebooks](https://jupyter.org/), as the output need not be in pure ASCII/Unicode. Just replace your `from tqdm import tqdm>` with`from tqdm.auto import tqdm` (Note: you might need to install `ipywidgets`) This makes `tqdm` automatically figure out if it is able to output the fancy progress bar.

This is how it looks like:

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-13-at-10.17.49-AM-1024x181.png)

![Image]({{ site.baseurl }}/assets/images/2024/04/Screen-Shot-2024-04-13-at-10.17.52-AM-1024x181.png)

### 6. More information

Find more information about `tqdm` here:

**Documentation** –[ https://tqdm.github.io/ ](< https://tqdm.github.io/ >)

**Github** – <https://github.com/tqdm/tqdm>
