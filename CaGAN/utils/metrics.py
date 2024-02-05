import numpy as np
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy_ALLClass(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Acc

    def Pixel_Precision_ALLClass(self):
        Pr=np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return Pr

    def F1_ALLClass(self):
        Acc=self.Pixel_Accuracy_ALLClass()
        Pr=self.Pixel_Precision_ALLClass()
        F1=(2*Acc*Pr)/(Acc+Pr)
        return F1

    def F1_MEANClass(self):
        Acc=self.Pixel_Accuracy_ALLClass()
        Pr=self.Pixel_Precision_ALLClass()
        F1=(2*Acc*Pr)/(Acc+Pr)
        F1_mean=np.nanmean(F1)
        return F1_mean

    def Class_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def plot_confusion_matrix(self, i_iter):
        plt.imshow(self.confusion_matrix,interpolation='nearest',cmap=plt.cm.Paired)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks=np.arange(self.num_class)
        plt.xticks(tick_marks,tick_marks)
        plt.yticks(tick_marks,tick_marks)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig("./%d.png"%i_iter)

if __name__ =="__main__":
    F1 = Evaluator.F1_ALLClass()
    F1_mean = Evaluator.F1_MEANClass()
    IoU = Evaluator.Class_Intersection_over_Union()
    Acc = Evaluator.Pixel_Accuracy()
    Acc_class = Evaluator.Pixel_Accuracy_Class()
    mIoU = Evaluator.Mean_Intersection_over_Union()
    FWIoU = Evaluator.Frequency_Weighted_Intersection_over_Union()
    self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
    self.writer.add_scalar('val/F1[0]', F1[0], epoch)
    self.writer.add_scalar('val/F1[1]', F1[1], epoch)
    self.writer.add_scalar('val/F1[2]', F1[2], epoch)
    self.writer.add_scalar('val/F1[3]', F1[3], epoch)
    self.writer.add_scalar('val/F1[4]', F1[4], epoch)
    self.writer.add_scalar('val/F1[5]', F1[5], epoch)
    self.writer.add_scalar('val/F1_mean', F1_mean, epoch)
    self.writer.add_scalar('val/IoU[0]', IoU[0], epoch)
    self.writer.add_scalar('val/IoU[1]', IoU[1], epoch)
    self.writer.add_scalar('val/IoU[2]', IoU[2], epoch)
    self.writer.add_scalar('val/IoU[3]', IoU[3], epoch)
    self.writer.add_scalar('val/IoU[4]', IoU[4], epoch)
    self.writer.add_scalar('val/IoU[5]', IoU[5], epoch)
    self.writer.add_scalar('val/mIoU', mIoU, epoch)
    self.writer.add_scalar('val/Acc', Acc, epoch)
    self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
    self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
    print("F1[0]:{}, F1[1]:{}, F1[2]:{}, F1[3]: {}, F1[4]: {}, F1[5]: {}, F1_mean: {}".format(F1[0], F1[1], F1[2], F1[3], F1[4], F1[5], F1_mean))
    print("IoU[0]:{}, IoU[1]:{}, IoU[2]:{}, IoU[3]: {}, IoU[4]: {}, IoU[5]: {}, mIoU: {}".format(IoU[0], IoU[1], IoU[2], IoU[3], IoU[4], IoU[5], mIoU))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))





