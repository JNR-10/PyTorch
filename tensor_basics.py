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
# ? For init with 0s
x = torch.zeros((3,3)) # similar for ones
# ? Initililize values from a uniform distribution (b/w 0 and 1)
x = torch.rand((3,3))
# ? For identity matrix
x = torch.eye(5,5)
# ? Like range function in python
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # will fill the values as per specified
# also possible with uniform distribution b/w a range
x = torch.empty(size=(1,5)).uniform_(0, 1)
# ? Diagonal matrix of 3x3 (this preserves other non-diagronal values as they are)
x = torch.diag(torch.ones(3))

# ! How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4) # int64 by default
x = tensor.bool()
x = tensor.short() # int16   
x = tensor.long() # int64
x = tensor.half() # float16
x = tensor.float() # float32
x = tensor.double() # float64

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
z1 = torch.add(x,y)
z1 = x + y

# ? Subtraction
z1 = x-y

# ? Division
z = torch.true_divide(x,y)
# element-wise division if they are of same dimension
# if y is scalar it would be a scalar division (all elements with a single given integer)

# ? In-place Operations
t = torch.zeros(3)

t.add_(x) # _ indicates the function is applied as an in-place function
t += x # t = t + x, this is not in-place this creates a new variable

# ? Exponentiation
z = x.pow(2)
z = x**2 # tensor([2., 4., 6.])

# ? Simple Comparision
z = x > 0 # tensor([True, True, True])
z = x < 0 # tensor([False, False, False])

# ? Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3 size
x3 = x1.mm(x2) # equivalent operation

# ? Matrix Exponentiation
matrix_exp = torch.rand(5,5)
print(
    matrix_exp.matrix_power(3)
) # is same as matrix_exp (mm) matrix_exp (mm) matrix_exp

# ? Element-wise multiplication
z = x * y

# ? Dot - Product (Element-wise multiplication and then sum)
z = torch.dot(x, y)

# ? Batch-Matrix multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # Will be shape: (b x n x p)

# ! Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.ones((1, 5))
z = (
    x1 - x2 # subtraction of a vector from a mtrix does not make sense mathematically but here it does!
) # ? Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
# ? This is called Broadcasting, when one entity broadens it's dimension to mathc the other element
z = (
    x1 ** x2
) # This is also an example of Broadcasting in PyTorch

# ! Other Useful tensor operations

sum_x = torch.sum(
    x, dim=0 # Sum of x across dim=0 (which is the only dim in our case), sum_x = 6
    # ? to specify which dimension it should sum over, usefule when 'x' is a matrix
)
values, indices = torch.max(x, dim=0) # Can also do x.max(dim=0) for most of the functions
# ? This will return the values and indices of maximum nad minimum values in matrix
values, indices = torch.min(x, dim=0) # Can also do x.min(dim=0)
abs_x = torch.abs(x) # ? Returns x where abs function has been applied to every element
z = torch.argmax(x, dim=0)
# ? This would do the same thig as above except it would just return the index
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0) # mean requires x to be float
z = torch.eq(x, y)  # in this case z = [False, False, False]
# ? Element wise comparison
sorted_y, indices = torch.sort(y, dim=0, descending=False)
# ? This returns sorted 'y' as well as the indices that we need to swap in order to make it sorted

z = torch.clamp(x, min=0, max=10)
# ? Basically this is like a bounding function any value exceeding the range will be set ot the upp/low limits of the range
# ? All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function i.e when max is not given)
# ? If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool) # ? True/False values
z = torch.any(x) # will return True, can also do x.any() instead of torch.any(x)
z = torch.all(
    x
) # will return False (since not all are True), can also do x.all() instead of torch.all()

# """
# * Tensor Indexing
# """

batch_size = 10
features = 25
x = torch.rand((batch_size, features)) 
# print(x)

# ? Get the features of the first example
print(x[0]) # shape [25], this is same as doing x[0,:]

# ? Get the first feature for all examples
print(x[:, 0]) # shape [10]

# ? # For example: Want to access third example in the batch and the first ten features
print(x[2, 0:10]) # shape: [10]

# ? # For example we can use this to, assign certain elements
x[0, 0] = 100

# ? Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices]) # x[indices] = [2, 5, 8]

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) # Gets second row fifth column and first row first column (because of ,0)

# ? More advanced Indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  # will be [0, 1, 9] ( | => 'or' operation)
print(x[x.remainder(2) == 0]) # will be [0, 2, 4, 6, 8]

# ? Useful operations for indexing
print(
    torch.where(x > 5, x, x * 2)
) # gives [0, 2, 4, 6, 8, 10, 6, 7, 8, 9], all values x > 5 yield x, else x*2
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()
print(
    x.ndimension()
) # The number of dimensions, in this case 1. if x.shape is 5x5x5 ndim would be 3
x = torch.arange(10)
print(
    x.numel() # ? Count number of elements in x
) # The number of elements in x (in this case it's trivial because it's just a vector)

# """
# * Tensor Reshaping
# """

x = torch.arange(9)

# x_3x3 = x.view(3, 3)

# x_3x3 = x.reshape(3, 3)

# y = x_3x3.t()
# print(
#     y.is_contiguous()
# )
# print(y.contiguous().view(9))

# x1 = torch.rand(2, 5)
# x2 = torch.rand(2, 5)
# print(torch.cat((x1, x2), dim=0).shape)
# print(torch.cat((x1, x2), dim=1).shape)

# z = x1.view(-1)

# batch = 64
# x = torch.rand((batch, 2, 5))
# z = x.view(
#     batch, -1
# )

# z = x.permute(0, 2, 1)

# z = torch.chunk(x, chunks=2, dim=1)
# print(z[0].shape)
# print(z[1].shape)

# x = torch.arange(
#     10
# )
# print(x.unsqueeze(0).shape)
# print(x.unsqueeze(1).shape)

# x = torch.arange(10).unsqueeze(0).unsqueeze(1)
# z = x.squeeze(1)