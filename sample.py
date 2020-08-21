import os
import os.path
import utils.config as config
import models.ddpm as ddpm
import utils.utils as utils


def modify_commandline_options(parser):
    '''
        Options only for sampling
    '''
    parser.add_argument('--phase', type=str, default='sample', help='Sampling Phase')
    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
    parser.add_argument('--results_dir', type=str, default='outputs/results/', help='saves results here.')
    parser.set_defaults(print_freq=1)


if __name__ == '__main__':
    # option setup
    cfg = config.Config()
    modify_commandline_options(cfg.parser)
    opt = cfg.parse()

    # model setup
    model = ddpm.DDPModel(opt)
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # save options in the checkpoints directory
    cfg.save()

    # save directory setup
    save_dir = os.path.join(opt.results_dir, opt.name)  # save all the images to save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('Directory created: %s' % save_dir)

    # sampling
    total_iters = 0                # the total number of training iterations

    for test in range(opt.num_test):
        outputs = model.sample()
        utils.save_images(outputs, [os.path.join(save_dir, 'sample_%d.png' % (total_iters+n)) for n in range(opt.batch_size)])
        total_iters += opt.batch_size
        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            print('sampled %d images in %s' % (total_iters, opt.results_dir))
