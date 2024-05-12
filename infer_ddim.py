import torch
import data as Data
import model as Model
import argparse
import logging
import numpy as np
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/diff-if-ivf_val.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    torch.manual_seed(1)
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0

    idx = 0

    Time_mat = np.zeros(len(val_loader)+1)

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data_collect in enumerate(val_loader):
        idx += 1
        name = str(val_data_collect[1][0]).replace(".jpg", "")
        val_data = val_data_collect[0]

        diffusion.feed_data(val_data)
        diffusion.test_ddim(continous=True)

        visuals = diffusion.get_val_current_visuals()

        vis_img = Metrics.tensor2img(visuals['vis'])  # uint8
        ir_img = Metrics.tensor2img(visuals['ir'])  # uint8
        x0_img = Metrics.tensor2img(visuals['x0'])  # uint8
        fake_img = Metrics.tensor2img(visuals['DDPM'])  # uint8
        fake_single = Metrics.tensor2img(visuals['DDPM'][-1].unsqueeze(0))  # uint8
        denoise_x0_img = Metrics.tensor2img(visuals['denoise_x0'])  # uint8
        img_full = Metrics.tensor2img(visuals['img_full'])  # uint8

        fuse_img = Metrics.mergy_Y_RGB_to_YCbCr(fake_single, img_full)

        sr_img_mode = 'fuse'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['DDPM']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            Metrics.save_img(
                fuse_img, '{}/{}.png'.format(result_path, name))
