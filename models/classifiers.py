import jittor as jt
import jittor.nn as nn


def renorm(x, p, dim, maxnorm):
    """
    Renormalizes input tensor x along the specified dimension.

    Args:
        x (Tensor): Input tensor.
        p (int): The order of norm.
        dim (int): Dimension along which to compute the norm.
        maxnorm (float): Maximum norm for renormalization.

    Returns:
        Tensor: Renormalized tensor along the specified dimension.

    Example:
        >>> import jittor as jt
        >>> x = jt.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> renorm(x, p=2, dim=1, maxnorm=1e-5)
    """
    # Compute the norm of x along the specified dimension
    x_norm = jt.norm(x, p=p, dim=dim)
    
    # Create a mask indicating elements with norm greater than maxnorm
    mask = x_norm > maxnorm
    
    # Expand the norm tensor to the same shape as x
    x_norm_expanded = x_norm.unsqueeze(1).broadcast(x.shape)
    
    # Apply renormalization where mask is True, otherwise keep original values
    x_renorm = jt.where(mask.unsqueeze(1).broadcast(x.shape), x / x_norm_expanded, x)
    
    return x_renorm


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, head_weight=None, dtype=None):
        super().__init__()
        if head_weight is None:
            weight = nn.Parameter(jt.empty(num_classes, feat_dim, dtype=dtype))
            weight.uniform_(-1, 1)
            self.weight = renorm(weight, 2, 1, 1e-5)
        else:
            self.weight = nn.Parameter(head_weight)

    @property
    def dtype(self):
        return self.weight.dtype

    def execute(self, x):
        raise NotImplementedError


class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, head_weight=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, head_weight, dtype)
        self.scale = scale

    def execute(self, x):
        x = jt.normalize(x, dim=-1)
        weight = jt.normalize(self.weight, dim=-1)
        return nn.linear(x, weight) * self.scale

