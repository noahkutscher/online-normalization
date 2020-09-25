import warnings

import torch
import torch.nn as nn

from online_norm_pytorch import _C


class LayerScalingK(nn.Module):
    r"""
    doc
    """

    __constants__ = ['eps']

    def __init__(self, eps=1e-05, **kwargs):
        super(LayerScalingK, self).__init__()
        self.eps = eps

        class LS(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                (out, scale, ) = _C.layer_scaling_fwd(input.contiguous(), self.eps)
                ctx.save_for_backward(out, scale,)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                out, scale, = ctx.saved_tensors
                (grad_in, ) = _C.layer_scaling_bwd(grad_out.contiguous(), out, scale)
                return grad_in

        self.ls = LS.apply

    def extra_repr(self):
        return f'eps={self.eps}'

    def forward(self, input):
        return self.ls(input)


def main():
    ls = LayerScalingK(1e-32)
    x = torch.randn(2, 3, 4, 5) + .2623
    x.requires_grad = True
    x_sanity = x.clone().detach()
    x_sanity.requires_grad = True
    dy = torch.randn(2, 3, 4, 5)
    y = ls(x)
    y.backward(dy)

    y_sanity = x_sanity / ((x_sanity * x_sanity).mean((1, 2, 3), keepdim=True) + 1e-32).sqrt()
    y_sanity.backward(dy)
    print(y.isclose(y_sanity).all())
    print(x.grad.isclose(x_sanity.grad).all())

if __name__ == '__main__':
    main()
