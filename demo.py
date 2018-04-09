import torch
import torch.nn as nn
import torch.autograd as autograd

# m = nn.Conv1d(410, 140, kernel_size=3, padding=1)
# input = autograd.Variable(torch.randn(98, 410, 100))
# output = m(input)
# print(output.shape)

tt = autograd.Variable(torch.randn(64,300,1,100))
m = nn.MaxPool2d((1,3), stride=1)
output = m(tt)
print(output.shape)

# m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(1, 1), padding=(0, 0))
#  # non-square kernels and unequal stride and with padding and dilation
# # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# input = autograd.Variable(torch.randn(20, 16, 50, 100))
# output = m(input)
# print(output.shape)