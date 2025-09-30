from torch import nn

from .Layers import PositionalEmbeddingLayer, Encoder, MLP


class STPE(nn.Module):
    def __init__(self, opt):
        super(STPE, self).__init__()
        self.PE = PositionalEmbeddingLayer(opt.d_model)
        self.Encoder = Encoder(opt)
        self.MLP = MLP(opt)

    def forward(self, x):
        # x is of shape [batch_size, shot, temporal_scale, feature_dimension]
        b, k, t, d = x.shape
        x = x.view(b * k, t, d)
        # x = self.PE(x)
        x = self.Encoder(x)
        x = self.MLP(x)
        x = x.view(b, k, t, -1)
        return x
