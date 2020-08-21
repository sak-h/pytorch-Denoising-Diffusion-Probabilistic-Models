import os
import os.path
import time
import utils.config as config
import utils.dataset as dataset
import models.ddpm as ddpm
import utils.visualizer as visualizer
import utils.utils as utils


def modify_commandline_options(parser):
    '''
        Options only for training
    '''
    parser.add_argument('--phase', type=str, default='train', help='training Phase')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--log_dir', type=str, default='outputs/log', help='log Directory')
    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true', help='save models by each iter')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=1000, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')


if __name__ == '__main__':
    # option setup
    cfg = config.Config()
    modify_commandline_options(cfg.parser)
    opt = cfg.parse()

    # dataset setup
    dataset = dataset.DatasetLoader(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # model setup
    model = ddpm.DDPModel(opt)
    model.setup(opt)

    # save options in the checkpoints directory
    cfg.save()

    # traiing

    # create a logging file to store training losses
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.train()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display current result on tensorboard
                model.compute_visuals(total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                utils.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, log_name)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_denoise_model(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_denoise_model('latest')
            model.save_denoise_model(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
