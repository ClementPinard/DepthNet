from __future__ import division
import torch.nn as nn
from models.utils import conv, deconv, predict_depth, post_process_depth, adaptative_cat, init_modules


class DepthNet(nn.Module):

    def __init__(self, batch_norm=False, with_confidence=False, clamp=False, depth_activation=None):
        super(DepthNet, self).__init__()

        self.clamp = clamp
        if depth_activation == 'elu':
            self.depth_activation = lambda x: nn.functional.elu(x) + 1
        else:
            self.depth_activation = depth_activation

        self.conv1   = conv(  6,   32, stride=2, batch_norm=batch_norm)
        self.conv2   = conv( 32,   64, stride=2, batch_norm=batch_norm)
        self.conv3   = conv( 64,  128, stride=2, batch_norm=batch_norm)
        self.conv3_1 = conv(128,  128,           batch_norm=batch_norm)
        self.conv4   = conv(128,  256, stride=2, batch_norm=batch_norm)
        self.conv4_1 = conv(256,  256,           batch_norm=batch_norm)
        self.conv5   = conv(256,  256, stride=2, batch_norm=batch_norm)
        self.conv5_1 = conv(256,  256,           batch_norm=batch_norm)
        self.conv6   = conv(256,  512, stride=2, batch_norm=batch_norm)
        self.conv6_1 = conv(512,  512,           batch_norm=batch_norm)

        self.deconv5 = deconv(512, 256, batch_norm=batch_norm)
        self.deconv4 = deconv(513, 128, batch_norm=batch_norm)
        self.deconv3 = deconv(385,  64, batch_norm=batch_norm)
        self.deconv2 = deconv(193,  32, batch_norm=batch_norm)

        self.predict_depth6 = predict_depth(512, with_confidence)
        self.predict_depth5 = predict_depth(513, with_confidence)
        self.predict_depth4 = predict_depth(385, with_confidence)
        self.predict_depth3 = predict_depth(193, with_confidence)
        self.predict_depth2 = predict_depth( 97, with_confidence)

        self.upsampled_depth6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_depth5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_depth4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_depth3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        init_modules(self)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out6                = self.predict_depth6(out_conv6)
        depth6 = post_process_depth(out6, clamp=self.clamp, activation_function=self.depth_activation)
        depth6_up           = self.upsampled_depth6_to_5(out6)
        out_deconv5         = self.deconv5(out_conv6)

        concat5     = adaptative_cat(out_conv5, out_deconv5, depth6_up)
        out5        = self.predict_depth5(concat5)
        depth5 = post_process_depth(out5, clamp=self.clamp, activation_function=self.depth_activation)
        depth5_up   = self.upsampled_depth5_to_4(out5)
        out_deconv4 = self.deconv4(concat5)

        concat4     = adaptative_cat(out_conv4, out_deconv4, depth5_up)
        out4        = self.predict_depth4(concat4)
        depth4 = post_process_depth(out4, clamp=self.clamp, activation_function=self.depth_activation)
        depth4_up   = self.upsampled_depth4_to_3(out4)
        out_deconv3 = self.deconv3(concat4)

        concat3     = adaptative_cat(out_conv3, out_deconv3, depth4_up)
        out3        = self.predict_depth3(concat3)
        depth3 = post_process_depth(out3, clamp=self.clamp, activation_function=self.depth_activation)
        depth3_up   = self.upsampled_depth3_to_2(out3)
        out_deconv2 = self.deconv2(concat3)

        concat2     = adaptative_cat(out_conv2, out_deconv2, depth3_up)
        out2        = self.predict_depth2(concat2)
        depth2 = post_process_depth(out2, clamp=self.clamp, activation_function=self.depth_activation)

        if self.training:
            return [depth2, depth3, depth4, depth5, depth6]
        else:
            return depth2