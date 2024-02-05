import torch
import torch.nn as nn
import numpy as np
from dataloaders.datasets import potsdam, vaihingen
from torch.utils.data import DataLoader
import torch.nn.functional as F


# from options.test_options import TestOptions


def class_mask(args, model, iter, source_num_classes, target_num_classes, batch_size):
    model.eval()
    mask_source = np.zeros((source_num_classes, 512, 512))
    #mask_source1 = torch.FloatTensor(source_num_classes, 512, 512)
    mask_target = np.zeros((target_num_classes, 512, 512))
    #mask_target1 = torch.FloatTensor(source_num_classes, 512, 512)

    source_features = torch.FloatTensor(source_num_classes, source_num_classes, 512, 512).cuda()
    source_features1 = torch.FloatTensor(batch_size, source_num_classes, source_num_classes, 512, 512).cuda()
    source_features3 = torch.FloatTensor(source_num_classes, batch_size, source_num_classes, 512, 512)

    target_features = torch.FloatTensor(target_num_classes, target_num_classes, 512, 512).cuda()
    target_features1 = torch.FloatTensor(batch_size, target_num_classes, target_num_classes, 512, 512).cuda()
    target_features3 = torch.FloatTensor(target_num_classes, batch_size, target_num_classes, 512, 512)

    c_s1 = torch.zeros(source_num_classes, batch_size, source_num_classes, 16, 16).cuda()
    c_t1 = torch.zeros(target_num_classes, batch_size, target_num_classes, 16, 16).cuda()
    c_t2 = torch.zeros(target_num_classes, batch_size, target_num_classes, 16, 16).cuda()

    # data
    source_set = potsdam.Potsdam(args, split='train', max_iters=None)
    target_set = vaihingen.Vaihingen(args, split='train', max_iters=None)
    sourceloader = DataLoader(source_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
    targetloader = DataLoader(target_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
    count = torch.zeros(6)

    # source
    for ii, sample in enumerate(sourceloader):
        images_source = sample['image']
        if args.cuda:
            images_source = images_source.cuda()
        with torch.no_grad():
            _, _, source_feature = model(images_source)
        for jj in range(source_feature.size()[0]):
            source_output = nn.functional.softmax(source_feature[jj], dim=1)
            source_output = source_output.data.cpu().numpy()
            source_label, source_prob = np.argmax(source_output, axis=0), np.max(source_output, axis=0)
            # source_prob = torch.squeeze(source_prob)
            source_l = np.zeros((512, 512), dtype=np.float32)
            a, b = source_label.shape

            for i in range(source_num_classes):
                N_S = 0
                for m in range(a):
                    for n in range(b):
                        if source_label[m, n] == i:
                            source_l[m, n] = 1
                            N_S += 1
                        else:
                            source_l[m, n] = 0
                if np.all(source_l == 0):
                    mask_source[i] = source_l
                else:
                    mask_source[i] = source_l + np.true_divide(source_l,N_S)
                mask_source1 = torch.from_numpy(mask_source[i])
                # get 6 classes
                mask_source1 = mask_source1.type_as(source_feature)
                if args.cuda:
                    mask_source1 = mask_source1.cuda()
                source_features[i] = mask_source1 * source_feature[jj]
            source_features1[jj] = source_features
        
        
        for i in range(source_num_classes):
            if torch.all(source_features1[:,i] == 0):
                c_s1[i] = torch.zeros(batch_size, 6, 16, 16)
            else:
                source_features3[i] = source_features1[:,i]
                c_s1[i] = F.interpolate(source_features3[i], size=[16,16], mode='bilinear', align_corners=True)
        if ii == 0:
            c_s = torch.mean(torch.stack([c_s1.clone(), c_s1.clone()]), 0)
        else:
            c_s = torch.mean(torch.stack([c_s, c_s1.clone()]), 0)
        print(c_s.shape)
        print('c_s:%d'%ii)
    print(c_s.shape)
    # target
    for ii, sample in enumerate(targetloader):
        images_target = sample['image']
        if args.cuda:
            images_target = images_target.cuda()
        with torch.no_grad():
            _, _, target_feature = model(images_target)
        for jj in range(target_feature.size()[0]):
            target_output = nn.functional.softmax(target_feature[jj], dim=1)
            target_output = target_output.data.cpu().numpy()
            target_label, target_prob = np.argmax(target_output, axis=0), np.max(target_output, axis=0)
            target_l = np.zeros((512, 512), dtype=np.float32)
            c, d = target_label.shape
    
            for i in range(target_num_classes):
                N_T = 0
                for m in range(c):
                    for n in range(d):
                        if target_label[m, n] == i:
                            target_l[m, n] = 1
                            N_T += 1
                        else:
                            target_l[m, n] = 0
                if np.all(target_l == 0):
                    mask_target[i] = target_l
                else:
                    mask_target[i] = target_l + np.true_divide(target_l,N_T)
                mask_target1 = torch.from_numpy(mask_target[i])
                # get 6 classes
                mask_target1 = mask_target1.type_as(target_feature)
                if args.cuda:
                    mask_target1 = mask_target1.cuda()
                target_features[i] = mask_target1 * target_feature[jj]
            target_features1[jj] = target_features.clone()

        
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        u=0.8
        T = 1.0/(1.5 + np.exp(-u*(iter+1)))

        for i in range(target_num_classes):
            if torch.all(target_features1[:,i] == 0):
                c_t1[i] = torch.zeros(batch_size, 6, 16, 16)
            else:
                target_features3[i] = target_features1[:,i]
                c_t1[i] = F.interpolate(target_features3[i], size=[16,16], mode='bilinear', align_corners=True)
            p_tl = cos(c_t1[i].clone(), c_s[i].clone())
            p_t = torch.mean(p_tl)
            if p_t >= T:
                if count[i] == 0:
                    c_t2[i] = torch.mean(torch.stack([c_t1[i].clone(), c_t1[i].clone()]), 0)
                else:
                    c_t2[i] = torch.mean(torch.stack([c_t2[i], c_t1[i].clone()]), 0)               
                count[i] = count[i] + 1
        print(count)
    c_t = c_t2.clone()
    print(c_t.shape)

    return c_s, c_t
