# plate

This is a simple implementation of holographic reduced representation by Plate (1993). The idea behind it is that you can recursively encode vector representations through circular convolution. A nice property of circular convolution is that it is invertible by involution, as long as you know one of the constituents of the representation.

# Example

```python3
import numpy as np
from plate import circular_convolution, decode

dog, bite = np.random.normal(size=(2, 100))

vec = circular_convolution(dog, bite)
b = decode(bite, vec)

# b should be more similar to dog than to other vectors.

```
