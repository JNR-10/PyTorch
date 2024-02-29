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
t = torch.zeros(3)
print(t)

t.add_(x)
t += x
print(t)

z = x.pow(2)
z = x**2
print(z)

z = x > 0 # tensor([True, True, True])
z = x < 0 # tensor([False, False, False])

x1 = torch.rand([2, 5])
x2 = torch.rand([5, 3])
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

matrix_exp = torch.rand(5,5)
print(
    matrix_exp.matrix_power(3)
)

print(x * y)

z = torch.dot(x, y)

batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm)

x1 = torch.rand((5, 5))
x2 = torch.ones((1, 5))
z = (
    x1 - x2
)
z = (
    x1 ** x2
)

sum_x = torch.sum(
    x, dim=0
)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(
    x
)

"""
* Tensor Indexing
"""

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)

print(x[:, 0].shape)

print(x[2, 0:10].shape)

x[0, 0] = 100

x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) 

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  # will be [0, 1, 9]
print(x[x.remainder(2) == 0])

print(
    torch.where(x > 5, x, x * 2)
)
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()
print(
    x.ndimension()
) 
x = torch.arange(10)
print(
    x.numel()
)