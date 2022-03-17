import torch
import torch.nn as nn
import torch.nn.functional as func

def get_convblock(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batchnorm=True):
    conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding, stride=stride)
    if batchnorm:
        layers = [conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def make_up_conv(in_channels,out_channels, kernel_size=2): #, padding=1):
    upscale = nn.Upsample(scale_factor=2, mode='bilinear')
    zeropad = nn.ZeroPad2d((0,1,0,1)) # pad only right and bottom to keep dimension after 2x2 conv
    conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=0)
    return nn.Sequential(upscale, zeropad, conv)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, init_weights=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        unet_in_channels = in_channels

        # con blocks labeled counterclockwise of typical UNet illustration
        self.conv1_1 = get_convblock(unet_in_channels,64,kernel_size=3)
        self.conv1_2 = get_convblock(64,64,kernel_size=3) #,stride=2)

        self.conv2_1 = get_convblock(64,128,kernel_size=3)
        self.conv2_2 = get_convblock(128,128,kernel_size=3)

        self.conv3_1 = get_convblock(128,256,kernel_size=3)
        self.conv3_2 = get_convblock(256,256,kernel_size=3)

        self.conv4_1 = get_convblock(256,512,kernel_size=3)
        self.conv4_2 = get_convblock(512,512,kernel_size=3)

        self.conv5_1 = get_convblock(512,1024,kernel_size=3)
        self.conv5_2 = get_convblock(1024,1024,kernel_size=3)

        self.upconv5 = make_up_conv(1024,512)

        self.conv6_1 = get_convblock(1024,512,kernel_size=3)
        self.conv6_2 = get_convblock(512,512,kernel_size=3)

        self.upconv6 = make_up_conv(512,256)

        self.conv7_1 = get_convblock(512,256,kernel_size=3)
        self.conv7_2 = get_convblock(256,256,kernel_size=3)

        self.upconv7 = make_up_conv(256,128)

        self.conv8_1 = get_convblock(256,128,kernel_size=3)
        self.conv8_2 = get_convblock(128,128,kernel_size=3)

        self.upconv8 = make_up_conv(128,64)

        self.conv9_1 = get_convblock(128,64,kernel_size=3)
        self.conv9_2 = get_convblock(64,64,kernel_size=3,stride=2)

        zeropad = nn.ZeroPad2d((0,1,0,1)) # pad only right and bottom to keep dimension after 2x2 conv
        final_conv = nn.Conv2d(64,num_classes,kernel_size=2)
        self.classifier = nn.Sequential(zeropad,final_conv)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        unet_input = x

        _, _, in_height, in_width = x.shape

        c1_1 = self.conv1_1(unet_input)
        c1_2 = self.conv1_2(c1_1)
        m1 = func.max_pool2d(c1_2,kernel_size=2, stride=2, padding=0, ceil_mode=True)

        c2_1 = self.conv2_1(m1)
        c2_2 = self.conv2_2(c2_1)
        m2   = func.max_pool2d(c2_2,kernel_size=2, stride=2, padding=0,ceil_mode=True)

        c3_1 = self.conv3_1(m2)
        c3_2 = self.conv3_2(c3_1)
        m3   = func.max_pool2d(c3_2,kernel_size=2, stride=2, padding=0,ceil_mode=True)

        c4_1 = self.conv4_1(m3)
        c4_2 = self.conv4_2(c4_1)
        m4   = func.max_pool2d(c4_2,kernel_size=2, stride=2, padding=0,ceil_mode=True)

        c5_1 = self.conv5_1(m4)
        c5_2 = self.conv5_2(c5_1)

        u5  = self.upconv5(c5_2)
        cu5 = torch.hstack([u5,c4_2])

        c6_1 = self.conv6_1(cu5)
        c6_2 = self.conv6_2(c6_1)

        u6  = self.upconv6(c6_2)
        cu6 = torch.hstack([u6,c3_2])

        c7_1 = self.conv7_1(cu6)
        c7_2 = self.conv7_2(c7_1)

        u7  = self.upconv7(c7_2)
        cu7 = torch.hstack([u7,c2_2])

        c8_1 = self.conv8_1(cu7)
        c8_2 = self.conv8_2(c8_1)

        u8  = self.upconv8(c8_2)
        cu8 = torch.hstack([u8,c1_2])

        c9_1 = self.conv9_1(cu8)
        c9_2 = self.conv9_2(c9_1)

        features = func.interpolate(c9_2, size=(in_height,in_width), mode='bilinear')
        return self.classifier(features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
