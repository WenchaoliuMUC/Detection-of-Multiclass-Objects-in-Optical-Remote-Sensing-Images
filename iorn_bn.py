import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd.variable import Variable

class ORBatchNorm2d(Module):

    def __init__(self, num_features, nOrientation, eps=1e-5, momentum=0.1, affine=True):
        super(ORBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.nOrientation = nOrientation
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        batch_size, channels, h, w = input.size()

        input_reshaped = input.view(batch_size, channels//self.nOrientation, h*self.nOrientation, w)

        result = F.batch_norm(
            input_reshaped, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

        return result.view(batch_size, channels, h, w)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))