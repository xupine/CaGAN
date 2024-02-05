import torch
import torch.nn as nn
from torch import Tensor


def CrossEntropyLoss(prediction, target):
    n, c, h, w = prediction.size()
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255, size_average=True)
    criterion = criterion.cuda()
    loss = criterion(prediction, target.long())
    loss /= n
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # type: Tensor
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


# single layer
def MMD_S(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    num_class = int(source.size()[0])
    loss_a = 0.0
    loss = 0.0
    for ii in range(num_class):
        source1 = source[ii]
        target1 = target[ii]
        source1 = source1.view(source1.size(0), -1)
        target1 = target1.view(target1.size(0), -1)
        kernels = guassian_kernel(source1.clone(), target1.clone(), kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        batch_size = int(source1.size()[0])
        for i in range(batch_size):
            s1, s2 = i, (i+1)%batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
            loss1 = loss / float(batch_size)
        loss_a += loss1
    return loss_a/num_class

def MMD_DS(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    num_class = int(source.size()[0])
    loss_a = 0.0
    loss = 0.0
    for ii in range(num_class):
        for jj in range(ii+1,num_class):
                source1 = source[ii]
                target1 = target[jj]
                source1 = source1.view(source1.size(0), -1)
                target1 = target1.view(target1.size(0), -1)
                kernels = guassian_kernel(source1.clone(), target1.clone(), kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
                batch_size = int(source1.size()[0])
                for i in range(batch_size):
                    s1, s2 = i, (i+1)%batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    loss += kernels[s1, s2] + kernels[t1, t2]
                    loss -= kernels[s1, t2] + kernels[s2, t1]
                    loss1 = loss / float(batch_size)
                loss_a += loss1
    return loss_a/((num_class*(num_class-1))/2)



# mutil-layers
def MMD_M(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, None]):
    num_class = 6
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss = 0
    for i in range(num_class):
        s1, s2 = i, i
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss



if __name__ == "__main__":
    a = torch.rand(6, 2, 1, 6, 6).cuda()
    b = torch.rand(6, 2, 1, 6, 6).cuda()
    m = torch.cat((a, b), dim=1)
    print(m.shape)
    print(m.narrow(1, 0, int(m.size(1)/2)).shape)
    print(MMD_DS(m.narrow(1, 0, int(m.size(1)/2)), m.narrow(1, int(m.size(1)/2), int(m.size(1)/2))).item())
