class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'vaihingen':
            return './CVPR_dataset/Vaihingen/'
        elif dataset == 'potsdam':
            return './CVPR_dataset/Potsdam/'
        if dataset == 'paviaU_S':
            return './CVPR_dataset/PaviaU_S/'
        elif dataset == 'paviaU_T':
            return './CVPR_dataset/PaviaU_T/'
        elif dataset == 'panchromatic':
            return './CVPR_dataset/panchromatic/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
