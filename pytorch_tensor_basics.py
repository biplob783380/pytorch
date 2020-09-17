import torch  # importing pytorch library


# Converting a list into a tensor:
my_list = [1,2,3,4,5,6] # this is list
my_tensor = torch.tensor(my_list) # this is converted into tensor data
# my_tensor = torch.tensor(my_list, dtype=torch.float32, device='cpu', requires_grad=True)  # custom set properties
print(my_tensor) # print the tensor
print(my_tensor.dtype) # print datadype of tensor (eg: float32, float64, int8 etc)
print(my_tensor.device) # print the device where this tensor is present (device: GPU/CPU)
print(my_tensor.shape) # print shape of a tensor
print(my_tensor.requires_grad)


# check if GPU available
print(torch.cuda.is_available()) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# quick initialize tenosrs:
my_tensor = torch.empty(size=(3,4)) # not initialized matrix
my_tensor = torch.ones(size=(3,4)) # matrix of ones
my_tensor = torch.zeros(size=(3,4)) # matrix of zeros
my_tensor = torch.eye(3) # identity matrix
my_tensor = torch.rand(size=(3,4)) # random value tensor
my_tensor = torch.randn(size=(3,4)) # random value tensor including -ve values
my_tensor = torch.arange(start=2, end=15, step=2) # create array 
my_tensor = torch.linspace(start=.1, end=2, steps=6) # create array with length 6
my_tensor = torch.empty(size=(3,4)).normal_(mean=0, std=1) # random value with mean=0 and std=1
my_tensor = torch.empty(size=(3,4)).uniform_(0,1) # random value with min=0 and max=1
my_tensor = torch.diag(torch.ones(3)) # another way to create identity matrix



# convert datatype of tensors (float, int, boolean)
my_tensor = torch.arange(4) # default dtype
my_tensor.bool() # boolean dtype
my_tensor.short() # int16
my_tensor.long() # int64
my_tensor.half() # float16
my_tensor.float() # float32
my_tensor.double() # float64



# numpy array conversion
import numpy as np
np_array = np.zeros((5,5))
my_tensor = torch.from_numpy(np_array)
np_array = my_tensor.numpy()
my_tensor


## tensor math operation
a = torch.tensor([1.0,2,3,4])
b = torch.tensor([5,6,7,8])

# addition
c = torch.add(a, b)
c = a+b
c = a.add_(2) # add 2

# substraction
c = a-b

# Division
c = a/b

# power
c = a.pow(2)
c = a**2

# matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,5))
torch.mm(x1,x2)
x1.mm(x2)

# matrix exponensial
torch.rand(3,3).matrix_power(2)

# dot product
torch.dot(torch.tensor([1,2,3]),torch.tensor([3,4,5]))

# More tensor operations
torch.sum(torch.rand(3, 4), dim=0) # sum on that dim
torch.argmax(torch.rand(3,4), dim=0) # returns index values only
torch.max(torch.rand(3,4), dim=0) # return max value and index of given dim
torch.min(torch.rand(3,4), dim=0) # return min
torch.abs(torch.randn(3,4)) # return all absolute values
torch.mean(torch.rand(3,4).float(), dim=0) # return mean
torch.eq(torch.rand(3,4), torch.rand(3,4)) # elementwise check for similarity between two matrix
torch.sort(torch.tensor([4,3,2,5]), dim=0, descending=False) # returns values and indeces; True to dec, False to asc.
torch.clamp(torch.tensor([4,3,2,5]), min=3) # set minimum to 3
torch.any(torch.tensor([0,0,1,0]).bool()) # is any of these are true
torch.all(torch.tensor([3,4,1,0]).bool()) # is all of these are true



x = torch.rand(10, 15)
x[:]  # get all value
x[: 6] # get 0 to before 6th index excluding 6th
x[:, 4] # from all of 1st dim select only 4th index value of second dim
x[2, 0:3] # from 2th index values returns 0 to 3 excluding 3th index;
x[2, 0:3] = 1000 # assigning 100 to these


x = torch.arange(10)
x[(x>6) | (x<2)] # print both of conditions values
x[(x>6) & (x<9)] # both condition is need to be true
x[x.remainder(2)==0] # all even elements
x[x%2==0] # all even elements [same as above]

torch.where(x>5) # return where x>5
torch.where(x>5, x, x+4) # if x>5 then return x else x+5
torch.tensor([1,1,2,2,3,3]).unique() # return unique values
torch.rand(3,4,3).ndimension() # return dimension count
torch.rand(3,4,3).numel() # returns number of elements


# Reshaping
random = torch.rand(3,4)
random.reshape(6,2)
random.view(6,2)
random.t() # transpose
random.t().contiguous().view(12) # use of view when got error

# concat : join on existing dimension
torch.cat((torch.rand(2,3), torch.rand(2,3)), dim=1)

# swipe dimensions
x = torch.rand(2,3,5)
x.permute(0, 2, 1) # 0 in its position, 2 in 1's position and 1 is on 2's position

# unsqueeze/ unsqueeze
torch.rand([1,4,3]).squeeze().unsqueeze(dim=0).shape



