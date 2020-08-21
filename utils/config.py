import os
import os.path
import argparse
from argparse import ArgumentParser
import yaml
import torch


class Config:
    def __init__(self):
        self.parser = ArgumentParser()
        # Common options
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--config_file', type=str, default='', help='config default file')
        self.parser.add_argument('--checkpoints_dir', type=str, default='outputs/checkpoints', help='Checkpoints Directory')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of filters in the first conv layer')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--dropout', action='store_true', help='use dropout')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--block_size', type=int, default=1, help='input block size')
        self.parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self.parser.add_argument('--beta_schedule', type=str, default='linear', help='beta interpolation method')
        self.parser.add_argument('--beta_start', type=int, default=0.0001, help='start beta value')
        self.parser.add_argument('--beta_end', type=int, default=0.02, help='end beta value')
        self.parser.add_argument('--num_timesteps', type=int, default=1000, help='# of timesteps')
        self.parser.add_argument('--loss_type', type=str, default='noisepred', help='loss prediction policy. [noisepred]')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--verbose', action='store_true', help='verbose mode')

    def parse(self):
        args = self.parser.parse_args()

        # set gpu ids
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])

        # load from config file
        if args.config_file != '':
            print('update default options by a config file: %s' % args.config_file)
            with open(args.config_file) as file:
                yaml_args = yaml.load(file, Loader=yaml.FullLoader)
            for k, v in yaml_args.items():
                default = self.parser.get_default(k)
                if v != default:
                    if hasattr(args, k):
                        setattr(args, k, v)

        assert isinstance(args.gpu_ids, list)

        self.print(args)

        self.args = args
        return args

    def print(self, opt):
        """Print options

        It will print both current options and default values(if different).
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save(self):
        """Save options

        It will save both current options and default values(if different) on CHECKPOINTS_DIR/NAME/PHASE+'_options.yaml'.
        """
        save_filename = os.path.join(self.args.checkpoints_dir, self.args.name, self.args.phase+'_options.yaml')
        print('save the config to %s' % save_filename)
        with open(save_filename, 'w') as file:
            yaml.dump(vars(self.args), file)
