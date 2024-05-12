'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    dataset = dataset_opt['dataset']
    if dataset == "MFNet":
        from data.ivf_dataset import MFNet_Dataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    resolution=dataset_opt['resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'])
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
    elif dataset == "Harvard":
        from data.mif_dataset import Harvard_Dataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    resolution=dataset_opt['resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'])
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
    elif dataset == "Test_ivf":
        from data.test_ivf_dataset import Test_Dataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    resolution=dataset_opt['resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'])
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
    elif dataset == "Test_mif":
        from data.test_mif_dataset import Test_Dataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    resolution=dataset_opt['resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'])
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
    else:
        raise 'the dataset type is wrong.'
    return dataset
