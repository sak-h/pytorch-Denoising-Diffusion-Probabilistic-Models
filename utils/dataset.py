import os
import os.path
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class DatasetLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.dataset = BasicDataset(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
        self.max_dataset_size = opt.max_dataset_size
        self.batch_size = opt.batch_size

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data


class BasicDataset(data.Dataset):
    """This dataset class"""

    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.tif', '.TIF', '.tiff', '.TIFF',
    ]

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        #super(BasicDataset).__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/train'

        self.paths = sorted(self.make_dataset(self.dir, opt.max_dataset_size))   # load images from '/path/to/data/train'
        self.size = len(self.paths)  # get the size of dataset A
        self.input_nc = opt.input_nc       # get the number of channels of input image
        self.transform = self.get_transform(opt, grayscale=(self.input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains images and paths
            images (tensor)       -- images
            paths (str)    -- image paths
        """
        path = self.paths[index % self.size]  # make sure index is within then range
        img = Image.open(path)
        img = img.convert('RGB')
        img = self.transform(img)

        return {'images': img, 'paths': path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size

    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in BasicDataset.IMG_EXTENSIONS)

    @staticmethod
    def make_dataset(dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if BasicDataset.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    @staticmethod
    def get_transform(opt, grayscale=False, method=Image.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, method))

        if 'crop' in opt.preprocess:
            transform_list.append(transforms.RandomCrop(opt.crop_size))

        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
