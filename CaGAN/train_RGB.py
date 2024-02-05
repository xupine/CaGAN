import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from Class_RGB import class_source, class_target
from class_mask_RGB import class_mask
from dataloaders.datasets import potsdam, vaihingen
from model.deeplab import *
from model.discriminator import FCDiscriminator
from model.sync_batchnorm.replicate import patch_replication_callback
from scripts.loss import Losses
from scripts.loss_MMD import MMD_S, MMD_DS
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from utils.saver_source import Saver_source
from utils.saver_target import Saver_target
from utils.saver import Saver

from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from scripts.mmd import mix_rbf_mmd2, mix_rbf_mmd2d

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver_source = Saver_source(args)
        self.saver_target = Saver_target(args)
        self.saver = Saver_target(args)


        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        self.summary_source = TensorboardSummary(self.saver_source.experiment_dir)
        self.writer_source = self.summary_source.create_summary()

        self.summary_target = TensorboardSummary(self.saver_target.experiment_dir)
        self.writer_target = self.summary_target.create_summary()

        # DATALOADERS
        # source
        self.source_train_set = potsdam.Potsdam(args, split='train', max_iters=args.max_iters * args.source_batch_size)
        self.source_val_set = potsdam.Potsdam(args, split='val', max_iters=None)
        self.source_num_class = self.source_train_set.NUM_CLASSES
        self.source_train_loader = DataLoader(self.source_train_set, batch_size=args.source_batch_size, shuffle=True, drop_last=True,
                                              num_workers=args.workers, pin_memory=True)
        self.source_val_loader = DataLoader(self.source_val_set, batch_size=args.source_batch_size, shuffle=False, drop_last=True,
                                            num_workers=args.workers, pin_memory=True)
        # target
        self.target_train_set = vaihingen.Vaihingen(args, split='train',max_iters=args.max_iters * args.target_batch_size)
        self.target_val_set = vaihingen.Vaihingen(args, split='val', max_iters=None)
        self.target_num_class = self.target_train_set.NUM_CLASSES
        self.target_train_loader = DataLoader(self.target_train_set, batch_size=args.target_batch_size, shuffle=True, drop_last=True,
                                              num_workers=args.workers, pin_memory=True)
        self.target_val_loader = DataLoader(self.target_val_set, batch_size=args.target_batch_size, shuffle=False, drop_last=True,
                                            num_workers=args.workers, pin_memory=True)
        # SEGMENTATION model
        model = DeepLab(num_classes=self.source_num_class,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        # DISCRIMINATOR model
        # high_feature level
        d_h = FCDiscriminator(num_classes=self.source_num_class)
        #d_hc = FCDiscriminator(num_classes=1)
        # low_feature level
        d_l = FCDiscriminator(num_classes=self.source_num_class)
        #d_lc = FCDiscriminator(num_classes=1)

        # Define Optimizer
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        # discriminators' optimizers
        optimizer_d_h = optim.Adam(d_h.parameters(), lr=args.lr_D,
                                   betas=(0.9, 0.99))
        #optimizer_d_hc = optim.Adam(d_hc.parameters(), lr=args.lr_D,
                                    #betas=(0.9, 0.99))
        optimizer_d_l = optim.Adam(d_l.parameters(), lr=args.lr_D,
                                   betas=(0.9, 0.99))
        #optimizer_d_lc = optim.Adam(d_lc.parameters(), lr=args.lr_D,
                                    #betas=(0.9, 0.99))
        # Define Criterion
        self.criterion = Losses(num_class=self.source_num_class, weight=None, batch_average=True, ignore_index=255,
                           cuda=args.cuda, size_average=True)
        #self.seg = criterion.CrossEntropyLoss()
        #self.bce_loss = criterion.bce_loss()
        
        #self.symkl2d = criterion.Symkl2d_class()
        #self.domain_adv = criterion.bce_adv()

        self.model, self.d_h, self.d_l = model, d_h, d_l
        self.optimizer, self.optimizer_d_h, self.optimizer_d_l = optimizer, optimizer_d_h, optimizer_d_l

        # Define Evaluator
        self.evaluator_source = Evaluator(self.source_num_class)
        self.evaluator_target = Evaluator(self.target_num_class)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,args.epochs, args.num_steps)
        self.scheduler_D = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, args.num_steps)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            self.d_h = torch.nn.DataParallel(self.d_h, device_ids=self.args.gpu_ids)
            self.d_h = self.d_h.cuda()
            self.d_l = torch.nn.DataParallel(self.d_l, device_ids=self.args.gpu_ids)
            self.d_l = self.d_l.cuda()
            #self.d_hc = torch.nn.DataParallel(self.d_hc, device_ids=self.args.gpu_ids)
            #self.d_hc = self.d_hc.cuda()
            #self.d_lc = torch.nn.DataParallel(self.d_lc, device_ids=self.args.gpu_ids)
            #self.d_lc = self.d_lc.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_pred_source = 0.0
        self.best_pred_target = 0.0
        if args.resume is not None:
            #if not os.path.isfile(args.resume):
                #raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            #self.best_pred_source = checkpoint['best_pred']
            #self.best_pred_target = checkpoint['best_pred']
            self.c_ss = checkpoint['c_ss']
            self.c_tt = checkpoint['c_tt']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # labels for adversarial training
        self.source_label = 0
        self.target_label = 1

    def training(self, epoch):
        if self.args.resume is not None:
            if epoch == self.args.start_epoch:
                c_ss = self.c_ss
                c_tt = self.c_tt
            else:
                iter_e = self.args.num_steps * epoch
                c_ss, c_tt = class_mask(self.args, self.model, iter_e, self.source_num_class, self.target_num_class, self.args.class_batch_size)
        else:
            iter_e = self.args.num_steps * epoch
            c_ss, c_tt = class_mask(self.args, self.model, iter_e, self.source_num_class, self.target_num_class, self.args.class_batch_size)
        if self.args.cuda:
            c_ss, c_tt = c_ss.cuda(), c_tt.cuda()
        #c_ss = torch.randn(self.source_num_class, 1, 6, 16, 16,device='cuda')
        #c_tt = torch.randn(self.source_num_class, 1, 6, 16, 16,device='cuda')
        self.model.train()
        self.d_h.train()
        sourceloader_iter = enumerate(self.source_train_loader)
        targetloader_iter = enumerate(self.target_train_loader)
        for iter_i in tqdm(range(self.args.num_steps)):
            # reset optimizers
            self.optimizer.zero_grad()
            self.optimizer_d_h.zero_grad()
            #self.optimizer_d_hc.zero_grad()

            # adapt LR if needed
            self.scheduler(self.optimizer, iter_i, epoch, self.best_pred)
            self.scheduler(self.optimizer_d_h, iter_i, epoch, self.best_pred)
            #self.model.adjust_learning_rate(self.args, self.optimizer, i)
            #self.d_h.adjust_learning_rate(self.args, self.optimizer_d_h, i)
            #self.d_hc.adjust_learning_rate(self.args, self.optimizer_d_hc, i)

            # UDA Training
            # only train segnet. Don't accumulate grads in disciminators
            for param in self.d_h.parameters():
                param.requires_grad = False
            #for param in self.d_hc.parameters():
                #param.requires_grad = False
            # train on source
            _, batch = sourceloader_iter.__next__()
            sample = batch
            images_source, labels_source = sample['image'], sample['label']
            if self.args.cuda:
                images_source, labels_source = images_source.cuda(), labels_source.cuda()

            as1, as2, pred_src_main1 = self.model(images_source)
            pred_src_main1 = pred_src_main1/1.8
            loss_seg = self.criterion.CrossEntropyLoss(pred_src_main1, labels_source)
            loss_seg = 1.0*loss_seg
            loss_seg.backward()


            # adversarial training ot fool the discriminator
            _, batch = targetloader_iter.__next__()
            sample = batch
            images_target = sample['image']
            at_1, at_2, pred_tar_main = self.model(images_target)
            as_1, as_2, pred_src_main = self.model(images_source)
            target_features1 = class_target(self.args.cuda, pred_tar_main, self.target_num_class)
            
            # update c_t
            #target_features2 = torch.FloatTensor(self.target_num_classes, self.target_num_classes, 512, 512)
            #target_features3 = torch.FloatTensor(self.target_num_class, 1, self.target_num_class, 512, 512)

            c_tb = torch.zeros(self.target_num_class, self.args.target_batch_size, 6, 16, 16).cuda()
            c_tb1 = torch.zeros(self.target_num_class, self.args.target_batch_size, 6, 16, 16).cuda()
            # target


            for i in range(self.target_num_class):
                target_features3 = target_features1[:,i].clone()
                if torch.all(target_features3 == 0):
                    c_tb1[i] = torch.zeros(self.args.target_batch_size, 6, 16, 16)
                else:
                    c_tb1[i] = F.interpolate(target_features3, size=[16,16], mode='bilinear', align_corners=True)
            if iter_i == 0:
                c_tb = torch.mean(torch.stack([c_tb1.clone(), c_tb1.clone()]), 0)
                
                #c_tbb = c_tb.detach()
            else:
                c_tb = torch.mean(torch.stack([c_tb, c_tb1.clone()]), 0)
               
                #c_tbb = c_tb.detach()
            # calu CS distance
               
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            p_t1 = cos(c_tt, c_tb)
            c_t = torch.pow(p_t1, 2) * c_tb + (1 - torch.pow(p_t1, 2)) * c_tt
            c_tt = c_t.detach()

            #source
            source_features3 = torch.FloatTensor(self.source_num_class, self.args.source_batch_size, self.source_num_class, 512, 512)
            c_sb = torch.zeros(self.source_num_class, self.args.source_batch_size, 6, 16, 16).cuda()
            c_sb1 = torch.zeros(self.source_num_class, self.args.source_batch_size, 6, 16, 16).cuda()
            source_features1 = class_source(self.args.cuda, pred_src_main, self.source_num_class)
            
            for i in range(self.source_num_class):
                source_features3[i] = source_features1[:,i].clone()
                if torch.all(source_features3[i] == 0):
                    c_sb1[i] = torch.zeros(self.args.source_batch_size, 6, 16, 16)
                else:
                    c_sb1[i] = F.interpolate(source_features3[i], size=[16,16], mode='bilinear', align_corners=True)
            if iter_i == 0:
                c_sb = torch.mean(torch.stack([c_sb1, c_sb1]), 0)
                
                #c_sbb = c_sb.detach()
            else:
                c_sb = torch.mean(torch.stack([c_sb, c_sb1]), 0)

                #c_sbb = c_sb.detach()

            # calu CS distance
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            p_s = cos(c_ss.clone(), c_sb.clone())
            c_s = torch.pow(p_s, 2) * c_sb + (1 - torch.pow(p_s, 2)) * c_ss
            c_ss = c_s.detach()
            c_st = torch.mean(torch.stack([c_s, c_t]), 0)

            # cal class loss to fool discriminator
            #m = torch.cat((c_s, c_t), dim=1)
            #n1 = torch.cat((c_s, c_st), dim=1)
            #n2 = torch.cat((c_t, c_st), dim=1)
            #n3 = torch.cat((c_st, c_st), dim=1)

            loss_adv_c1i = self.criterion.Diff2d_class(c_s, c_t)
            loss_adv_c2i = self.criterion.Diff2d_class(c_s, c_st)
            loss_adv_c3i = self.criterion.Diff2d_class(c_t, c_st)
            loss_adv_cd = self.criterion.Diff2d_dclass(c_st, c_st)

            # cal attention plenty
            as_11 = torch.zeros(pred_tar_main.size()[0], 64, 64).cuda()
            at_11 = torch.zeros(pred_tar_main.size()[0], 64, 64).cuda()
            a_1 = torch.zeros(pred_tar_main.size()[0], 64, 64).cuda()

            as_22 = torch.zeros(pred_tar_main.size()[0], 64, 64).cuda()
            at_22 = torch.zeros(pred_tar_main.size()[0], 64, 64).cuda()
            a_2 = torch.zeros(pred_tar_main.size()[0], 64, 64).cuda()

            for j in range(pred_tar_main.size()[0]):
                loss_1 = 0
                ass_1 = as_1[j].clone()
                att_1 = at_1[j].clone()
                as_11[j] = torch.div(ass_1,torch.norm(ass_1,p=2))
                at_11[j] = torch.div(att_1,torch.norm(att_1,p=2))
                a_1[j] = as_11[j].clone() - at_11[j].clone()
                loss1 =  torch.norm(a_1[j].clone(),p=2)
                loss_1 += loss1
            for j in range(pred_tar_main.size()[0]):
                loss_2 = 0
                ass_2 = as_2[j].clone()
                att_2 = at_2[j].clone()
                as_22[j] = torch.div(ass_2,torch.norm(ass_2,p=2))
                at_22[j] = torch.div(att_2,torch.norm(att_2,p=2))
                a_2[j] = as_22[j].clone() - at_22[j].clone()
                loss2 = torch.norm(a_2[j].clone(),p=2)
                loss_2 += loss2
            loss_att = loss_1 + loss_2
           

            # calu. domai loss to fool discriminator
            d_out_main = self.d_h(F.softmax(pred_tar_main,dim=1))
            loss_adv_a = self.criterion.bce_adv(d_out_main, self.source_label)
            loss_adv_t = 0.001*(loss_adv_a) + 0.0002*loss_att + 0.0002*(loss_adv_c1i+loss_adv_c2i+loss_adv_c3i-loss_adv_cd)
            loss_adv_t.backward()
            

            # Train discriminator networks
            # enable training mode on discriminator networks
            for param in self.d_h.parameters():
                param.requires_grad = True
            #for param in self.d_hc.parameters():
                #param.requires_grad = True

            # train witn source
            # domain
            pred_src_main = pred_src_main.detach()
            d_out_main = self.d_h(F.softmax(pred_src_main,dim=1))

            loss_d_main_s = self.criterion.bce_adv_DS(d_out_main, self.source_label)
       
            #source_features2 = torch.FloatTensor(self.source_num_classes, self.source_num_classes, 512, 512)
            
            loss_d_sou = loss_d_main_s / 2
            loss_d_sou.backward()

            # train with target
            # domain
            pred_tar_main = pred_tar_main.detach()
            d_out_main = self.d_h(F.softmax(pred_tar_main,dim=1))
            loss_d_main_t = self.criterion.bce_adv_DT(d_out_main, self.target_label)
            
            loss_d_tar = loss_d_main_t / 2
            loss_d_tar.backward()
            self.optimizer.step()
            self.optimizer_d_h.step()

            #optimizer_d_hc.step()
            #output loss
            current1_losses = {'loss_seg': loss_seg.item(),
                              'loss_adv_t': loss_adv_t.item(),
                              'loss_d_sou': loss_d_sou.item(),
                              'loss_d_tar': loss_d_tar.item()}
            print(iter_i, current1_losses)

            current2_losses = {'loss_adv_a': loss_adv_a.item(),
                              'loss_att': loss_att.item(),
                              'loss_adv_c1i': loss_adv_c1i.item(),
                              'loss_adv_c2i': loss_adv_c2i.item(),
                              'loss_adv_c3i': loss_adv_c3i.item(),
                              'loss_adv_cd': loss_adv_cd.item()}
            print(iter_i, current2_losses)
            #Tensorboard
            self.writer.add_scalar('train/loss_seg', loss_seg.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_adv_t', loss_adv_t.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_d_sou', loss_d_sou.item(), iter_i + self.args.num_steps * epoch)
            self.writer.add_scalar('train/loss_d_tar', loss_d_tar.item(), iter_i + self.args.num_steps * epoch)
            sys.stdout.flush()


            #validation
            if iter_i % 500 == 0 and iter_i != 0:
                self.validation_source(iter_i,epoch,c_ss,c_tt)
                self.validation_target(iter_i,epoch,c_ss,c_tt)


        print(epoch, current1_losses)
        
        #save
        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            filename1 = 'checkpoint_model.pth.tar'
            filename2 = 'checkpoint_d_h.pth.tar'
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'c_ss' : c_ss,
                'c_tt' :c_tt,
                'c_t11' :c_t11,
            }, is_best, filename1)

            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.d_h.state_dict(),
                'optimizer': self.optimizer_d_h.state_dict(),
            }, is_best, filename2)

    def validation_source(self, i_iter, epoch, c_ss, c_tt):
        self.model.eval()
        self.evaluator_source.reset()
        tbar_source = tqdm(self.source_val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar_source):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                _, _, output = self.model(image)
            loss = self.criterion.CrossEntropyLoss(output, target)
            test_loss += loss.item()
            tbar_source.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator_source.add_batch(target, pred)
            
        F1 = self.evaluator_source.F1_ALLClass()
        F1_mean = self.evaluator_source.F1_MEANClass()
        IoU = self.evaluator_source.Class_Intersection_over_Union()
        Acc = self.evaluator_source.Pixel_Accuracy()
        Acc_class = self.evaluator_source.Pixel_Accuracy_Class()
        mIoU = self.evaluator_source.Mean_Intersection_over_Union()
        FWIoU = self.evaluator_source.Frequency_Weighted_Intersection_over_Union()
        F1_mean1 = (F1[1] + F1[2] + F1[3] + F1[4] + F1[5]) / 5.0
        mIoU1 = (IoU[1] + IoU[2] + IoU[3] + IoU[4] + IoU[5]) / 5.0
        self.writer_source.add_scalar('val_source/total_loss_epoch', test_loss, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1[0]', F1[0], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1[1]', F1[1], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1[2]', F1[2], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1[3]', F1[3], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1[4]', F1[4], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1[5]', F1[5], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1_mean', F1_mean1, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1_mean', F1_mean, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/IoU[0]', IoU[0], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/IoU[1]', IoU[1], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/IoU[2]', IoU[2], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/IoU[3]', IoU[3], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/IoU[4]', IoU[4], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/IoU[5]', IoU[5], i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/F1_mean', mIoU1, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/mIoU', mIoU, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/Acc', Acc, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/Acc_class', Acc_class, i_iter + self.args.num_steps * epoch)
        self.writer_source.add_scalar('val_source/fwIoU', FWIoU, i_iter + self.args.num_steps * epoch)
        file1 = "./Acc_source.txt"
        with open(file1, 'a', encoding='utf-8') as f:
            f.writelines(str(Acc) + '\n')
        file2 = "./Acc_class_source.txt"
        with open(file2, 'a', encoding='utf-8') as f1:
            f1.writelines(str(Acc_class) + '\n')
        file3 = "./F1_mean1_source.txt"
        with open(file3, 'a', encoding='utf-8') as f2:
            f2.writelines(str(F1_mean1) + '\n')
        file4 = "./mIoU1_source.txt"
        with open(file4, 'a', encoding='utf-8') as f3:
            f3.writelines(str(mIoU1) + '\n')


        print('Validation_source:')
        print("F1[0]:{}, F1[1]:{}, F1[2]:{}, F1[3]: {}, F1[4]: {}, F1[5]: {}, F1_mean: {}, F1_mean1: {}".format(F1[0],F1[1],F1[2],F1[3],F1[4],F1[5],F1_mean,F1_mean1))
        print("IoU[0]:{}, IoU[1]:{}, IoU[2]:{}, IoU[3]: {}, IoU[4]: {}, IoU[5]: {}, mIoU: {}, mIoU1: {}".format(IoU[0],IoU[1],IoU[2],IoU[3],IoU[4],IoU[5],mIoU,mIoU1))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU1
        if new_pred > self.best_pred_source:
            is_best = True
            filename = 'checkpoint_model_source.pth.tar'
            self.best_pred_source = new_pred
            self.saver_source.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred_source,
                'c_ss' : c_ss,
                'c_tt' :c_tt,
            }, is_best, filename)

    def validation_target(self, i_iter, epoch, c_ss, c_tt):
        self.model.eval()
        self.evaluator_target.reset()
        tbar_target = tqdm(self.target_val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar_target):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                _, _, output = self.model(image)
            loss = self.criterion.CrossEntropyLoss(output, target)
            test_loss += loss.item()
            tbar_target.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator_target.add_batch(target, pred)

        F1 = self.evaluator_target.F1_ALLClass()
        F1_mean = self.evaluator_target.F1_MEANClass()
        IoU = self.evaluator_target.Class_Intersection_over_Union()
        Acc = self.evaluator_target.Pixel_Accuracy()
        Acc_class = self.evaluator_target.Pixel_Accuracy_Class()
        mIoU = self.evaluator_target.Mean_Intersection_over_Union()
        FWIoU = self.evaluator_target.Frequency_Weighted_Intersection_over_Union()
        F1_mean1 = (F1[1] + F1[2] + F1[3] + F1[4] + F1[5]) / 5.0
        mIoU1 = (IoU[1] + IoU[2] + IoU[3] + IoU[4] + IoU[5]) / 5.0
        self.writer_target.add_scalar('val_target/total_loss_epoch', test_loss, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1[0]', F1[0], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1[1]', F1[1], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1[2]', F1[2], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1[3]', F1[3], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1[4]', F1[4], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1[5]', F1[5], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1_mean', F1_mean1, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1_mean', F1_mean, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/IoU[0]', IoU[0], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/IoU[1]', IoU[1], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/IoU[2]', IoU[2], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/IoU[3]', IoU[3], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/IoU[4]', IoU[4], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/IoU[5]', IoU[5], i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/F1_mean', mIoU1, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/mIoU', mIoU, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/Acc', Acc, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/Acc_class', Acc_class, i_iter + self.args.num_steps * epoch)
        self.writer_target.add_scalar('val_target/fwIoU', FWIoU, i_iter + self.args.num_steps * epoch)
        file1 = "./Acc_target.txt"
        with open(file1, 'a', encoding='utf-8') as f:
            f.writelines(str(Acc) + '\n')
        file2 = "./Acc_class_target.txt"
        with open(file2, 'a', encoding='utf-8') as f1:
            f1.writelines(str(Acc_class) + '\n')
        file3 = "./F1_mean1_target.txt"
        with open(file3, 'a', encoding='utf-8') as f2:
            f2.writelines(str(F1_mean1) + '\n')
        file4 = "./mIoU1_target.txt"
        with open(file4, 'a', encoding='utf-8') as f3:
            f3.writelines(str(mIoU1) + '\n')
        file5 = "./IoU0.txt"
        with open(file5, 'a', encoding='utf-8') as f4:
            f4.writelines(str(IoU[0]) + '\n')
        file6 = "./IoU1.txt"
        with open(file6, 'a', encoding='utf-8') as f5:
            f5.writelines(str(IoU[1]) + '\n')
        file7 = "./IoU2.txt"
        with open(file7, 'a', encoding='utf-8') as f6:
            f6.writelines(str(IoU[2]) + '\n')
        file8 = "./IoU3.txt"
        with open(file8, 'a', encoding='utf-8') as f7:
            f7.writelines(str(IoU[3]) + '\n')0.
        file9 = "./IoU4.txt"
        with open(file9, 'a', encoding='utf-8') as f8:
            f8.writelines(str(IoU[4]) + '\n')
        file10 = "./IoU5.txt"
        with open(file10, 'a', encoding='utf-8') as f9:
            f9.writelines(str(IoU[5]) + '\n')
            
        print('Validation_target:')
        print("F1[0]:{}, F1[1]:{}, F1[2]:{}, F1[3]: {}, F1[4]: {}, F1[5]: {}, F1_mean: {}, F1_mean1: {}".format(F1[0],F1[1],F1[2],F1[3],F1[4],F1[5],F1_mean,F1_mean1))
        print("IoU[0]:{}, IoU[1]:{}, IoU[2]:{}, IoU[3]: {}, IoU[4]: {}, IoU[5]: {}, mIoU: {}, mIoU1: {}".format(IoU[0],IoU[1],IoU[2],IoU[3],IoU[4],IoU[5],mIoU,mIoU1))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU1
        if new_pred > self.best_pred_target:
            is_best = True
            filename = 'checkpoint_model_target.pth.tar'
            self.best_pred_target = new_pred
            self.saver_target.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred_target,
                'c_ss' : c_ss,
                'c_tt' :c_tt,
            }, is_best, filename)

def main():
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    # data
    parser.add_argument('--source-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
								training (default: auto)')
    parser.add_argument('--target-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
								training (default: auto)')
    parser.add_argument('--class-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    # seg net
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    # optimizer
    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_D', type=float, default=1e-4, metavar='LR_D',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # LR
    parser.add_argument("--max-iters", type=int, default=7000, help="Max number of training steps.")
    parser.add_argument("--num-steps", type=int, default=7000, help="Number of training steps in every epoch.")

    # training  params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')

    # GPU
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default="0",
                        help='use which gpu to train, must be a \
						comma-separated list of integers only (default=0,1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #save checkpoint
    parser.add_argument("--save-pred-every", type=int, default=500, help="Save summaries and checkpoint every often.")
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.checkname is None:
        args.checkname = 'class-gan-' + str(args.backbone)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)


if __name__ == "__main__":
    main()
