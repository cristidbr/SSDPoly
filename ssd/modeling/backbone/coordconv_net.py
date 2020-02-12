import torch
import torch.nn as nn
import torch.nn.modules.conv as conv

from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    

class CoordConvNet( nn.Module ):
    def __init__( self, cfg, width_mult=1.0, inverted_residual_setting=None ):
        super().__init__()
       
        block = InvertedResidual
        input_channel = 32
        last_channel = 80

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 8, 1, 1],
                [6, 12, 2, 2],
                [6, 16, 3, 2],
                [6, 16, 4, 2],
                [6, 24, 3, 1],
                [6, 20, 3, 2],
                [6, 40, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [
            ConvBNReLU( 3, input_channel, stride=2 ),
            CoordConv2d( input_channel, input_channel, 3, stride = 1, with_r=True )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.extras = nn.ModuleList([
            InvertedResidual(last_channel, 64, 2, 0.2),
            InvertedResidual(64, 32, 2, 0.25),
            InvertedResidual(32, 32, 2, 0.5),
            InvertedResidual(32, 16, 2, 0.25)
        ])

        self.init_parameters()

    def init_parameters( self ):
        # weight initialization
        for m in self.modules():
            if isinstance( m, nn.Conv2d ):
                nn.init.kaiming_normal_( m.weight, mode='fan_out' )
                if m.bias is not None:
                    nn.init.zeros_( m.bias )
            elif isinstance( m, nn.BatchNorm2d ):
                nn.init.ones_( m.weight )
                nn.init.zeros_( m.bias )
            elif isinstance( m, nn.Linear ):
                nn.init.normal_( m.weight, 0, 0.01 )
                nn.init.zeros_( m.bias )

    def forward(self, x):
        features = []
        for i in range(15):
            x = self.features[i](x)
        features.append(x)

        for i in range(14, len(self.features)):
            x = self.features[i](x)
        features.append(x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        return tuple(features)

@registry.BACKBONES.register( 'coordconv' )
def coordconv( cfg, pretrained = False ):
    model = CoordConvNet( cfg )
    model_url = ''
    #if pretrained:
    #    model.init_from_pretrain(load_state_dict_from_url(model_url))

    return model