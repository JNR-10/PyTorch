import torch

"""
* Initializing Tensor
"""
device = "cuda" if torch.cuda.is_available() else "cpu" # default is cpu
# ! Creating a tensor
# ? So we are having 2 rows and 3 columns (2x3) in this tensor
# ? We can also set the type for this tensor
# ? Set the device the tensor should be on
# ? other paramerters like required_grad is essential for autograd (learnt later)
my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32,
                         device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# ! Other common initialization methods
# ? Creating an random(uninit) matrix of specified size (3x3)
x = torch.empty(size=(3,3))
print(x)
# ? For init with 0s
x = torch.zeros((3,3)) # similar for ones
print(x)
# ? Initililize values from a uniform distribution (b/w 0 and 1)
x = torch.rand((3,3))
print(x)
# ? For identity matrix
x = torch.eye(5,5)
print(x)
# ? Like range function in python
print(torch.arange(start=0, end=5, step=1))
print(torch.linspace(start=0.1, end=1, steps=10))
print(torch.empty(size=(1,5)).normal_(mean=0, std=1)) # will fill the values as per specified
# also possible with uniform distribution b/w a range
print(torch.empty(size=(1,5)).uniform_(0, 1))
# ? Diagonal matrix of 3x3 (this preserves other non-diagronal values as they are)
print(torch.diag(torch.ones(3)))

# ! How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4) # int64 by default
print(tensor)
print(tensor.bool())
print(tensor.short()) # int16   
print(tensor.long()) # int64
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

# ! Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()


"""
* Tendor math and computation operations
"""

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# ? Addition
z1 = torch.empty(3)
torch.add(x,y, out=z1)
print(z1)
print(torch.add(x,y))
print(x + y)

# ? Subtraction
print(x-y)

# ? Division
z = torch.true_divide(x,y)
print(z) # element-wise division if they are of same dimension
# if y is scalar it would be a scalar division (all elements with a single given integer)

# ? In-place Operations

