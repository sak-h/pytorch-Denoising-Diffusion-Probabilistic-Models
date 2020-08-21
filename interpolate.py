import os
import os.path
import utils.config as config
import utils.dataset as dataset
import models.ddpm as ddpm
import utils.utils as utils


def modify_commandline_options(parser):
    '''
        Options only for interpolation
    '''
    parser.add_argument('--phase', type=str, default='interpolate', help='Interpolation Phase')
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--num_test', type=float, default=float("inf"), help='how many test images to run')
    parser.add_argument('--results_dir', type=str, default='outputs/results/', help='saves results here')
    parser.add_argument('--mix_rate', type=float, default=0.5, help='mixing rate')
    parser.set_defaults(print_freq=1)


if __name__ == '__main__':
    # option setup
    cfg = config.Config()
    modify_commandline_options(cfg.parser)
    opt = cfg.parse()

    # dataset setup
    opt.batch_size = 2             # only support batch_size=2
    opt.serial_batches = True
    dataset = dataset.DatasetLoader(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

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

    for i, data in enumerate(dataset):
        if i>= opt.num_test:
            break
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        outputs = model.interpolate()
        utils.save_images(outputs, [os.path.join(save_dir, model.get_interpolate_filename())])
        total_iters += opt.batch_size
        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            print('interpolate %d images in %s' % (total_iters, opt.results_dir))
