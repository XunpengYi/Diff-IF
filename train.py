import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/diff-if-ivf.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

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

    best_psnr = 0

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data_collect in enumerate(val_loader):
                        idx += 1
                        name = str(val_data_collect[1][0])
                        val_data = val_data_collect[0]
                        diffusion.feed_data(val_data)
                        diffusion.test_ddim(continous=False)
                        visuals = diffusion.get_current_visuals()
                        vis_img = Metrics.tensor2img(visuals['vis'])  # uint8
                        ir_img = Metrics.tensor2img(visuals['ir'])  # uint8
                        x0_img = Metrics.tensor2img(visuals['x0'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['DDPM'])  # uint8
                        denoise_x0_img = Metrics.tensor2img(visuals['denoise_x0'])  # uint8
                        img_full = Metrics.tensor2img(visuals['img_full']) #uint8

                        fuse_img = Metrics.mergy_Y_RGB_to_YCbCr(fake_img, img_full)

                        Metrics.save_img(
                            fuse_img, '{}/{}.png'.format(result_path, name))

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d} Evaluation.>'.format(current_epoch, current_step))

                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, best_psnr_flag=False)

                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        logger.info('Saving best_psnr models and training states.')
                        diffusion.save_network(current_epoch, current_step, best_psnr_flag=True)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step, best_psnr_flag=False)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data_collect in enumerate(val_loader):
            idx += 1
            name = str(val_data_collect[1][0])
            val_data = val_data_collect[0]
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            vis_img = Metrics.tensor2img(visuals['vis'])  # uint8
            ir_img = Metrics.tensor2img(visuals['ir'])  # uint8
            x0_img = Metrics.tensor2img(visuals['x0'])  # uint8
            fake_img = Metrics.tensor2img(visuals['DDPM'])  # uint8

            Metrics.save_img(
                vis_img, '{}/{}_{}_vis.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                ir_img, '{}/{}_{}_ir.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                x0_img, '{}/{}_{}_x0.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = (Metrics.calculate_psnr(Metrics.tensor2img(visuals['DDPM'][-1]), vis_img)
                        + Metrics.calculate_psnr(Metrics.tensor2img(visuals['DDPM'][-1]), ir_img))/2
            eval_ssim = (Metrics.calculate_ssim(Metrics.tensor2img(visuals['DDPM'][-1]), vis_img)
                        + Metrics.calculate_ssim(Metrics.tensor2img(visuals['DDPM'][-1]), ir_img))/2

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))