class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'vaihingen':
            return '/media/user/新加卷/xupine_cvpr/CVPR_dataset/Vaihingen/'
        elif dataset == 'potsdam':
            return '/media/user/新加卷/xupine_cvpr/CVPR_dataset/Potsdam/'
        elif dataset == 'paviaU':
            return './CVPR_dataset/paviaU/'
        elif dataset == 'panchromatic':
            return './CVPR_dataset/panchromatic/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
