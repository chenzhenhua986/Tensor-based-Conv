import torch.nn as nn
import torch
import torch.nn.functional as F
from tensor_conv import StructuredInputLayer, TensorConvLayer2d, TensorConvLayer3d
from tensor_conv import TensorBatchNorm2d, TensorBatchNorm3d, TensorBatchNorm
from tensor_conv import TensorAdaptiveAvgPool2d, TensorMaxPool2d
from tensor_io import StructuredInputLayer, StructuredOutputLayer
from tensor_deconv import TensorTransposeConvLayer2d
from deeplab import DeepLabHead
from tconv_deeplab import TconvDeepLabHead
from fcn import FCNHead
from lraspp import lraspp_backbone, lraspp_backbone1, LRASPPHead
from torchvision.ops import StochasticDepth
from resnet import ResNet, Bottleneck
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, fcn_resnet50, fcn_resnet101, lraspp_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights
import timeit


class tconv_seg_head(nn.Module):
    def __init__(self, device, num_classes, dr_rate):
        super(tconv_seg_head, self).__init__()

        self.block1 = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
        )

        self.dropout = nn.Sequential(
            nn.Dropout(dr_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=num_classes, input_tensor=[13, 13], out_type=0),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.head(x)
        return x

class tconv4a_head(nn.Module):
    def __init__(self, device, num_classes, dr_rate):
        super(tconv4a_head, self).__init__()

        self.f0 = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
        )
        self.s1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s4 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block41 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s5 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block51 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s6 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block61 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s7 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block71 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s8 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block81 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s9 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block91 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.dropout = nn.Sequential(
            nn.Dropout(dr_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=num_classes, input_tensor=[17, 17], out_type=0),
        )

    def forward(self, x):
        x = self.f0(x)
        identity = self.s1(x)
        x11 = self.block11(x)+identity

        identity = self.s2(x11)
        x21 = self.block21(x11)+identity

        identity = self.s3(x21)
        x31 = self.block31(x21)+identity

#        identity = self.s4(x31)
#        x41 = self.block41(x31)+identity
#
#        identity = self.s5(x41)
#        x51 = self.block51(x41)+identity
#
#        identity = self.s6(x51)
#        x61 = self.block61(x51)+identity
#
#        identity = self.s7(x61)
#        x71 = self.block71(x61)+identity
#
#        identity = self.s8(x71)
#        x81 = self.block81(x71)+identity
#
#        identity = self.s9(x81)
#        x91 = self.block91(x81)+identity

        x = self.head(x31)
        return x

class tconv4a(nn.Module):
    def __init__(self, device):
        super(tconv4a, self).__init__()

        self.features = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[3, 1], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
        )
        self.shortcut1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.shortcut2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.shortcut3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.shortcut4 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block41 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s5 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block51 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s6 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block61 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s7 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block71 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s8 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block81 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.s9 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )
        self.block91 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[17, 17], output_tensor=[17, 17], out_type=2),
            TensorBatchNorm2d(17*17),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1) 
        x = x.unsqueeze(5)
        x = self.features(x)
        identity = self.shortcut1(x)
        x11 = self.block11(x)+identity

        identity = self.shortcut2(x11)
        x21 = self.block21(x11)+identity
 
        identity = self.shortcut3(x21)
        x31 = self.block31(x21)+identity

        identity = self.shortcut4(x31)
        x41 = self.block41(x31)+identity

        identity = self.s5(x41)
        x51 = self.block51(x41)+identity

        identity = self.s6(x51)
        x61 = self.block61(x51)+identity

        identity = self.s7(x61)
        x71 = self.block71(x61)+identity

        identity = self.s8(x71)
        x81 = self.block81(x71)+identity

        identity = self.s9(x81)
        x91 = self.block91(x81)+identity

        return x91

class tconv_seg_backbone(nn.Module):
    def __init__(self, device):
        super(tconv_seg_backbone, self).__init__()

        self.features = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[3, 1], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
        )
        self.shortcut1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.shortcut2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=2, stride_h=2, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.shortcut3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.shortcut4 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block41 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.s5 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block51 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.s6 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block61 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.s7 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block71 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.s8 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block81 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.s9 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )
        self.block91 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1) 
        x = x.unsqueeze(5)
        x = self.features(x)
        identity = self.shortcut1(x)
        x11 = self.block11(x)+identity

        identity = self.shortcut2(x11)
        x21 = self.block21(x11)+identity
 
        identity = self.shortcut3(x21)
        x31 = self.block31(x21)+identity

        identity = self.shortcut4(x31)
        x41 = self.block41(x31)+identity

        identity = self.s5(x41)
        x51 = self.block51(x41)+identity

        identity = self.s6(x51)
        x61 = self.block61(x51)+identity

        identity = self.s7(x61)
        x71 = self.block71(x61)+identity

        identity = self.s8(x71)
        x81 = self.block81(x71)+identity

        identity = self.s9(x81)
        x91 = self.block91(x81)+identity

        return x91

class tconv_backbone_tconv_head(nn.Module):
    def __init__(self, device, num_classes):
        super(tconv_backbone_tconv_head, self).__init__()

        #self.backbone = tconv4a(device)
        #self.classifier = tconv4a_head(device, num_classes, 0.5)
        self.backbone = tconv_seg_backbone(device)
        #self.classifier = tconv4a_head(device, num_classes, 0.5)
        self.classifier = tconv_seg_head(device, num_classes, 0.5)


    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)   
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)             
        return {"out": x}

class backbone_lraspp(nn.Module):
    def __init__(self, device):
        super(backbone_lraspp, self).__init__()


class classifier_lraspp(nn.Module):
    def __init__(self, device, num_classes):
        super(classifier_lraspp, self).__init__()


#2.2M parameters
class tconv_lraspp(nn.Module):
    def __init__(self, device, num_classes):
        super(tconv_lraspp, self).__init__()

        #self.backbone = lraspp_backbone(device)
        self.backbone = lraspp_backbone1(device)
        self.classifier = LRASPPHead(512, 512, num_classes, 128)


    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return {"out": x}
        #return x



class official_deeplab_resnet(nn.Module):
    def __init__(self, num_classes):
        super(official_deeplab_resnet, self).__init__()

        self.backbone = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], replace_stride_with_dilation=[False, True, True])
        self.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return {"out": x}

class head1(nn.Module):
    def __init__(self, device, input_num, dr_rate, num_classes):
        super(head1, self).__init__()

        self.f0 = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=8, out_channel=1, input_tensor=[16, 16], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
        )
        self.s1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.s2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.s3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.s4 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block41 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.dropout = nn.Sequential(
            nn.Dropout(dr_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.fcn_head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=num_classes, input_tensor=[11, 11], out_type=0),
        )

    def forward(self, x):
        x = x.view(-1, 8, 16, 16, 32, 32)
        x = x.permute(0, 1, 4, 5, 2, 3)
        x = self.f0(x)
        identity = self.s1(x)
        x11 = self.block11(x)+identity

        identity = self.s2(x11)
        x21 = self.block21(x11)+identity

        identity = self.s3(x21)
        x31 = self.block31(x21)+identity

        identity = self.s4(x31)
        x41 = self.block41(x31)+identity

        #x21 = self.dropout(x21)  64.8% [13, 13]
        #x31 = self.dropout(x31)   65.0% iter30, [13, 13]
        #x41 = self.dropout(x41)  # 64.0% iter30, [13, 13]
        #x41 = self.dropout(x41)  # 64.5% iter15, [11, 11]
        #x41 = self.dropout(x41)  # 65.4% iter30, [11, 11], no dropout
        x = self.fcn_head(x41)
        return x


class head4(nn.Module):
    def __init__(self, device, input_num, dr_rate, num_classes):
        super(head4, self).__init__()

        self.f0 = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=8, out_channel=1, input_tensor=[16, 16], output_tensor=[13, 13], out_type=2),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
        )
        self.s1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s4 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block41 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s5 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block51 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s6 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block61 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s7 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block71 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s8 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block81 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.s9 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.block91 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[13, 13], output_tensor=[13, 13], out_type=1),
            TensorBatchNorm2d(13*13),
        )
        self.dropout = nn.Sequential(
            nn.Dropout(dr_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.fcn_head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=num_classes, input_tensor=[13, 13], out_type=0),
        )

    def forward(self, x):
        x = x.view(-1, 8, 16, 16, 32, 32)
        x = x.permute(0, 1, 4, 5, 2, 3)
        x = self.f0(x)
        identity = self.s1(x)
        x11 = self.block11(x)+identity

        identity = self.s2(x11)
        x21 = self.block21(x11)+identity

        identity = self.s3(x21)  #pascal, [13, 13], 64.6 iter#30 || [17, 17], 64.0, iter#30, || [19, 19], 63.4, iter#30
        x31 = self.block31(x21)+identity

        identity = self.s4(x31)    #pascal, [13, 13] 64.7 iter#30 || [17, 17], 64.9, iter#54 || [18, 18], 63.8, iter#30
        x41 = self.block41(x31)+identity
        identity = self.s5(x41)  #pascal 64.2, [13, 13], iter#30||64,1, [11, 11], ite#30 || [17, 17], 64.4%, iter#24 || [18, 18], 64.3, iter#33 || [19, 19]
        x51 = self.block51(x41)+identity
        identity = self.s6(x51)  #pascal[11, 11], 64.1  #iter30 || [13, 13], 64.6, iter30 || [17, 17], 65.3%, iter#30 ||[18, 18] 65.1%, iter#33 || [19, 19], 63.5%, iter#30
        x61 = self.block61(x51)+identity
        identity = self.s7(x61) #pascal, [13, 13], 64.9% #iter33 || [17, 17], 65.3, iter#54 ||[18, 18], 64.0, iter#30   || [19, 19], 63.9, iter#33
        x71 = self.block71(x61)+identity 
        identity = self.s8(x71) #pascal #iter30, [13, 13], 64.2% || [17, 17], 64.5%, iter#30 ||[18, 18], 65.2, iter#54 || [19, 19],64.1 iter#33 
        x81 = self.block81(x71)+identity
        identity = self.s9(x81)       # pascal [19, 19] 64.5%, iter30 ||[17, 17], 63.8, iter#30 || [18, 18], 64.3, iter#33 ||[13, 13], 65.6%, iter#30
        x91 = self.block91(x81)+identity

        x = self.fcn_head(x91)
        return x

class head3(nn.Module):
    def __init__(self, device, input_num, dr_rate, num_classes):
        super(head3, self).__init__()

        self.f0 = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=8, out_channel=1, input_tensor=[16, 16], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
        )
        self.s1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s4 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block41 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s5 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block51 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s6 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block61 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s7 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block71 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s8 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block81 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.s9 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.block91 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[18, 18], output_tensor=[18, 18], out_type=2),
            TensorBatchNorm2d(18*18),
        )
        self.fcn_head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=num_classes, input_tensor=[18, 18], out_type=0),
        )

    def forward(self, x):
        x = x.view(-1, 8, 16, 16, 32, 32)
        x = x.permute(0, 1, 4, 5, 2, 3)
        x = self.f0(x)
        identity = self.s1(x)
        x11 = self.block11(x)+identity

        identity = self.s2(x11)
        x21 = self.block21(x11)+identity

        dentity = self.s3(x21)  
        x31 = self.block31(x21)+identity

#        identity = self.s4(x31)    
#        x41 = self.block41(x31)+identity
#        identity = self.s5(x41)  
#        x51 = self.block51(x41)+identity
#        identity = self.s6(x51)  
#        x61 = self.block61(x51)+identity
#        identity = self.s7(x61) 
#        x71 = self.block71(x61)+identity 
#        identity = self.s8(x71) 
#        x81 = self.block81(x71)+identity
#        identity = self.s9(x81) 
#        x91 = self.block91(x81)+identity

        x = self.fcn_head(x31)
        return x


class head2(nn.Module):
    def __init__(self, device, input_num, dr_rate, num_classes):
        super(head2, self).__init__()

        self.f0 = nn.Sequential(
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=8, out_channel=1, input_tensor=[16, 16], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
        )
        self.s1 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block11 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.s2 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block21 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.s3 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.block31 = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=3, kw=3, pad_w=1, pad_h=1, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
            nn.PReLU(),
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=1, input_tensor=[11, 11], output_tensor=[11, 11], out_type=2),
            TensorBatchNorm2d(11*11),
        )
        self.dropout = nn.Sequential(
            nn.Dropout(dr_rate),
        )
        self.relu = nn.ReLU(inplace=True)
        self.fcn_head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=1, out_channel=num_classes, input_tensor=[11, 11], out_type=0),
        )

    def forward(self, x):
        x = x.view(-1, 8, 16, 16, 32, 32)
        x = x.permute(0, 1, 4, 5, 2, 3)
        x = self.f0(x)
        identity = self.s1(x)
        x11 = self.block11(x)+identity

        identity = self.s2(x11)
        x21 = self.block21(x11)+identity

        identity = self.s3(x21)
        x31 = self.block31(x21)+identity

        x = self.fcn_head(x31)
        return x


class multi_classifier(nn.Module):
    def __init__(self, device, in_channels, num_classes):
        super(multi_classifier, self).__init__()
        modules = []
        modules.append(head4(device, in_channels, 0.5, num_classes))
        modules.append(head4(device, in_channels, 0.5, num_classes))
        self.convs = nn.ModuleList(modules)
        self.fcn_head = nn.Sequential(
            TensorConvLayer2d(device, kh=1, kw=1, pad_w=0, pad_h=0, stride_w=1, stride_h=1, in_channel=len(modules), out_channel=num_classes, input_tensor=[11, 11], out_type=0),
        )
    def forward(self, x):
        _res = []
        for conv in self.convs:
          _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.fcn_head(res)


class pretrained_backbone_tconv_head(nn.Module):
    def __init__(self, device, num_classes):
        super(pretrained_backbone_tconv_head, self).__init__()

        #self.backbone = nn.Sequential(*list(deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights, num_classes=num_classes).backbone.children())[:-3])
        #self.backbone = nn.Sequential(*list(deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, num_classes=num_classes).backbone.children()))
        #self.backbone = nn.Sequential(*list(deeplabv3_resnet101(num_classes=num_classes).backbone.children()))
        #self.backbone = nn.Sequential(*list(deeplabv3_resnet50(num_classes=num_classes).backbone.children()))
        self.backbone = nn.Sequential(*list(deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, num_classes=num_classes).backbone.children()))
        #self.backbone = nn.Sequential(*list(deeplabv3_mobilenet_v3_large(num_classes=num_classes).backbone.children()))
        #self.backbone = nn.Sequential(*list(deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT, num_classes=num_classes).backbone.children()))

        #self.classifier = TconvDeepLabHead(device, 2048, num_classes)
        #self.classifier = DeepLabHead(2048, num_classes)
        #self.classifier = DeepLabHead(960, num_classes)
        #self.classifier = FCNHead(2048, num_classes)
        #self.classifier = FCNHead(960, num_classes)
        #self.classifier = head(device, 2048, 0.5, num_classes)
        #self.classifier = head1(device, 2048, 0.5, num_classes) 
        #self.classifier = head2(device, 2048, 0.5, num_classes)
        self.classifier = head3(device, 2048, 0.5, num_classes)
        #self.classifier = head4(device, 2048, 0.1, num_classes)
        #self.classifier = multi_classifier(device, 2048, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.classifier(x)
        #print(x.size())
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return {"out": x}







