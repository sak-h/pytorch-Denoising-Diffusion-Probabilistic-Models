import os
from torch.utils.tensorboard import SummaryWriter


class Visualizer():
    """Tensorboard wrapper"""
    
    def __init__(self, opt):
        log_dir = os.path.join(opt.log_dir, opt.name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print('Directory created: %s' % log_dir)
        self.writer = SummaryWriter(log_dir)
        print('Visualiser created (log in %s)' % log_dir)

    def add_models(self, models=[]):
        for model in models:
            self.writer.add_graph(model['model'], model['input'])

    def add_values(self, values=[]):
        for idx, value in enumerate(values):
            label = value['name'] if 'name' in value else str(idx)
            x = value['x'] if 'x' in value else 0
            y = value['y'] if 'y' in value else 0
            self.writer.add_scalar(label, y, x)

    def add_images(self, images=[]):
        for idx, image in enumerate(images):
            if 'data' not in image:
                continue
            img = image['data']
            img = (img+1.)*0.5
            img[img<0] = 0
            img[img>1] = 1
            label = image['name'] if 'name' in image else str(idx)
            step = image['step'] if 'step' in image else 0
            self.writer.add_images(label, img, step)
