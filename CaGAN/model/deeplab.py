import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.aspp import build_aspp
from model.decoder import build_decoder
from model.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=8, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x,x1,x2 = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.sum(torch.pow(x1,2),1)
        x2 = torch.sum(torch.pow(x2,2),1)




        return x1, x2, x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()



    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.lr * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=8)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    _,output = model(input)
    print(output.size())


