import torch
from torch import nn
import numpy as np

def class_source(cuda, pred_src_main, source_num_classes):

	mask_source = np.zeros((source_num_classes,512,512))
	source_features = torch.FloatTensor(source_num_classes, source_num_classes, 512, 512).cuda()
	source_features1 = torch.FloatTensor(pred_src_main.size()[0], source_num_classes, source_num_classes, 512, 512).cuda()
	for ii in range(pred_src_main.size()[0]):
		source_output = nn.functional.softmax(pred_src_main[ii], dim=0)
		source_output = source_output.data.cpu().numpy()
		source_label, source_prob = np.argmax(source_output, axis=0), np.max(source_output, axis=0)
		source_l = np.zeros((512, 512), dtype=np.float32)
		a,b = source_label.shape
		
		for i in range(source_num_classes):
			N_S = 0
			for m in range(a):
				for n in range(b):
					if source_label[m,n] == i:
						source_l[m,n] = 1
						N_S += 1
					else:
						source_l[m,n] = 0
			if np.all(source_l == 0):
				mask_source[i] = source_l
			else:
				mask_source[i] = source_l + np.true_divide(source_l,N_S)
			mask_source1 = torch.from_numpy(mask_source[i])
			#get 6 classes
			mask_source1 = mask_source1.type_as(pred_src_main[ii])
			if cuda:
				mask_source1 = mask_source1.cuda()
			source_features[i] = mask_source1 * pred_src_main[ii]
		source_features1[ii] = source_features.clone()
	return source_features1


def class_target(cuda, pred_tar_main, target_num_classes):
	
	mask_target = np.zeros((target_num_classes,512,512))
	target_features = torch.FloatTensor(target_num_classes, target_num_classes, 512, 512).cuda()
	target_features1 = torch.FloatTensor(pred_tar_main.size()[0], target_num_classes, target_num_classes, 512, 512).cuda()


	for ii in range(pred_tar_main.size()[0]):
		target_output = nn.functional.softmax(pred_tar_main[ii], dim=1)
		target_output = target_output.data.cpu().numpy()
		target_label, target_prob = np.argmax(target_output, axis=0), np.max(target_output, axis=0)
		target_l = np.zeros((512,512),dtype=np.float32)
		c, d = target_label.shape
		for i in range(target_num_classes):
			N_T = 0
			for m in range(c):
				for n in range(d):
					if target_label[m,n] == i:
						target_l[m,n] = 1
						N_T += 1
					else:
						target_l[m,n] = 0
			if np.all(target_l == 0):
				mask_target[i] = target_l
			else:
				mask_target[i] = target_l + np.true_divide(target_l,N_T)
				
			mask_target1 = torch.from_numpy(mask_target[i])
			# get 6 classes
			mask_target1 = mask_target1.type_as(pred_tar_main[ii])
			if cuda:
				mask_target1 = mask_target1.cuda()
			target_features[i] = mask_target1 * pred_tar_main[ii]
		target_features1[ii] = target_features.clone()
	return target_features1