
## Robust Principle Component Analysis

Repository for algorithms built with Pytorch for robust PCA, along with various utility functions for dealing with images. 
* https://www.jmlr.org/papers/volume21/18-884/18-884.pdf - Sub gradient solver
* https://arxiv.org/pdf/0912.3599.pdf Robust Principle Component Analysis

Both solvers require Numpy and Pytorch.


## Example
This is a simple demo on video set that has been corrupted


```python
# Torch and Numpy are needed for RPCA
import torch
import numpy as np

# Imports for building that data set and showing results.
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image, display
import os
from os import chdir
# Import Local Files with scripts
import sys
from Pursuit import *
from SubGD import *
from Utils import *

import time
# Set Device to CUDA to run GPU
# This notebook is run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# call the image folder, the build function should 
image_folder = "MovedObject"
#set the directory
os.chdir()
```

### Build the dataset


```python
# pass the image folder to the build function
# start and end allow for users to only work with subsets of a whole folder of images. Here we only work it the portion that contains motion
data_frame_ = build_data_set(image_folder,
                            start=630,
                            end=820,
                            standardized = True )

# This function takes a sequence of frames and compiles into a gif
return_animated_gif(data_frame_, output_filename='ExampleGif.gif', frame_duration=5, image_size=(120, 160))

# Display the corrupted gif
from IPython.display import Image, display
display(Image(filename='ExampleGif.gif'))
```

    GIF saved successfully.
![Alt Text](https://github.com/Tomleahy12/Robust-PCA/blob/main/ExampleGif.gif)


## Call SubGD Solver



```python
# If the rank is unknown, pass a list of guesses and solve iteratively, then  store the results using the code that i give below
# we only run for a brief amount of time in this example.

ranks_ = [2]
sparse_results_ = []
for iteration, rank in enumerate(ranks_):
    start_time = time.time()
    subdd_model = SubGD(data_frame_, device_ = device, rank_ = rank)
    print(f"Rank for problem is: {rank}")
    L,S = subdd_model.fit(iterations = 1200,
                        tolerance = .1,
                        mu = .9, 
                        beta_ = .95,
                        gamma_ = .2, 
                        prints = True,
                        iter_prints = 100)
    # I suggest removing torch tensors from device before storing
    # best practice is to always clone the tensor.
    sparse = S.clone().detach().cpu().numpy()
    sparse_results_.append(sparse)
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Total duration: {duration_seconds:.2f} seconds")
```

    Rank for problem is: 2
     error rate: 0.6784954730833966 at iteration 100
    .
    .
    .
     error rate: 0.17368023565056204 at iteration 1100
    Total duration: 63.37 seconds
    

### Returns 


```python
# Extract from the list this way. If on GPU, use .detach().cpu().numpy(). Or just call sparse from prev cell
sparse_show = torch.tensor(sparse_results_[0]).numpy()

# Takes the top 10% of values and maximizes. Rest are 0. 
percentile_animated_gif(sparse_show, output_filename='ExampleRPCA1.gif',percentile_cutoff=95, frame_duration=5, image_size=(120, 160))

from IPython.display import Image, display
display(Image(filename='ExampleRPCA1.gif'))
```

    GIF saved successfully as ExampleRPCA1.gif.
    
 ![Alt Text](https://github.com/Tomleahy12/Robust-PCA/blob/main/ExampleRPCA1.gif)

### Example of Highly Corrupted Image 
```python
# we call a different data set building function that will also tamper with the images. 
data_frame_vc = build_corrupted_imageset(image_folder, start = 600, end=820,method = 'Remove', corrupt_param= .7, standardized=True )

# This function takes a sequence of frames and compiles into a gif
return_animated_gif(data_frame_, output_filename='ExampleGifcorrupted.gif', frame_duration=5, image_size=(120, 160))

# we display the corrupted gif
from IPython.display import Image, display
display(Image(filename='ExampleGifcorrupted.gif'))
```
    GIF saved successfully.
![Alt](https://github.com/Tomleahy12/Robust-PCA/blob/main/ExampleGifcorrupted.gif)
### Call the Solver
```python
# If the rank is unknown, pass a list of guesses and solve iteratively, and store the results using the code that i gave below
# we only run for a brief amount of time on local in this example.
ranks_ = [4]
sparse_results_ = []
for iteration, rank in enumerate(ranks_):
    start_time = time.time()
    subdd_model = SubGD(data_frame_, device_ = device, rank_ = rank)
    print(f"Rank for problem is: {rank}")
    L,S = subdd_model.fit(iterations = 5000,
                        tolerance = .1,
                        mu = .9, 
                        beta_ = .95,
                        gamma_ = .5, 
                        prints = True,
                        iter_prints = 10)
    sparsevc = S.clone().detach().cpu().numpy()
    sparse_results_.append(sparse)
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Total duration: {duration_seconds:.2f} seconds")
```
    Rank for problem is: 4
     error rate: 0.7242109180848748 at iteration 10
      .
      .
      .
     error rate: 0.7254877602446004 at iteration 4990
    Total duration: 358.61 seconds
### Returns  
```python
# Takes the top 10% of values and maximizes. Rest are 0. 
percentile_animated_gif(sparsevc, output_filename='ExampleRPCA.gif',percentile_cutoff=96, frame_duration=5, image_size=(120, 160))
from IPython.display import Image, display
display(Image(filename='ExampleRPCA.gif'))
```

    GIF saved successfully as ExampleRPCA.gif.
![Alt](https://github.com/Tomleahy12/Robust-PCA/blob/main/ExampleRPCA.gif)
