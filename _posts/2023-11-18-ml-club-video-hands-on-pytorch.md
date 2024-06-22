---
id: 67
title: 'ML Club Video: Hands-on PyTorch!'
date: '2023-11-18T23:16:43+00:00'

layout: post
guid: 'http://karthikvedula.com/?p=67'
permalink: /2023/11/18/ml-club-video-hands-on-pytorch/
categories:
    - 'Machine Learning'
    - 'ML Club'
    - Uncategorized
tags:
    - 'Machine Learning'
    - 'Neural Network'
    - Programming
    - PyTorch
---

In this ML Club session, we do a walk-through of how to code a simple neural network in the Python programming language. We will be using a deep-learning library called PyTorch to code a neural network that can recognize handwritten digits (MNIST dataset).

Here is a quick summary of the content we go over.

### Modules:

- The base class for all neural network modules in PyTorch is `torch.nn.Module`. When building a neural network, you typically create a custom class that inherits (OOP!) from `Module` and define the layers and operations in the `__init__` method and the forward pass in the `forward` method.

```python
import torch.nn as nn

# Example of a simple neural network with a linear layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=5)

    def forward(self, x):
        x = self.fc(x)
        return x
```

### Layers:

- Various neural network layers are available in the `torch.nn` module, such as `Linear` (fully connected layer), `Conv2d` (2D convolutional layer), `LSTM` (Long Short-Term Memory), and many others. These layers can be used to construct the architecture of your neural network.

### Activation Functions:

- Activation functions like `ReLU` (Rectified Linear Unit), `Sigmoid`, and `Tanh` are derived from `torch.nn.functional` Activation functions introduce non-linearity to the neural network, enabling it to learn more complex data.
- *Activation functions can also be included as a layer (e.g. `torch.nn.ReLU()` )*

### Sequential Container:

- The `torch.nn.Sequential` container allows you to sequentially compose a sequence of layers. This simplifies the process of creating and managing the architecture of a neural network.

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_sequential = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_sequential = nn.Sequential(
            nn.Linear(64*4*4, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 10)
        )
```

### Loss Functions:

- PyTorch provides a variety of loss functions in `torch.nn` for different tasks, such as classification (`CrossEntropyLoss`), regression (`MSELoss`), and more.
- Remember loss functions determine how “wrong” the model’s prediction is from the actual ground truth. This is then used for our **optimizer (see next section)**.

### Optimizers:

- The `torch.optim` module, provides various optimization algorithms such as SGD (Stochastic Gradient Descent), Adam, and RMSprop. These optimizers update the model parameters **given the loss** during the training process.

### Datasets and Dataloaders

- These classes are used for being able to load and iterate through data. Datasets and dataloaders also allow the specification of **batch size** and preprocessing tasks to take place prior to training time.

[Here is the link to the code used in the lecture that shows the above concepts]({{ site.baseurl }}/assets/images/2023/11/NNSimple-1.html)

**See the video lecture below**!

<iframe allow="autoplay" height="480" loading="lazy" src="https://drive.google.com/file/d/1vvfa-p-QzTb1I8wHqmc7tXE9-FkDVLrW/preview" width="640"></iframe>
