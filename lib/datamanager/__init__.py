import numpy as np
from lib.utils.util import ConstType
from lib.dataset import DataBank
from lib.datamanager.triplet_sampler import AllTripletSampler
from lib.datamanager.test_sampler import TestSampler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from lib.datamanager.transforms import GroupToPILImage, StackTensor

"""
类声明：加载数据
"""
class DataManager(ConstType):
    def __init__(self, name, root_dir, rawfiles_dir, split_id, use_flow, seed, minframes, num_workers, logger):
        self.seed = seed
        self.use_flow = use_flow
        self.num_workers = num_workers
        self.logger = logger
        self.dataset = DataBank(name, root_dir, rawfiles_dir, split_id, np.random.RandomState(self.seed), minframes=minframes, logger=self.logger)
        self.npr = np.random.RandomState(self.seed)

        if self.use_flow:
            assert not self.dataset.is_image_dataset and self.dataset.shape[-1] == 5

    def set_transform(self, transform):
        transform = Compose([GroupToPILImage(use_flow=self.use_flow)] + transform + [StackTensor()])
        self.dataset.set_transform(transform)

    def set_train_generator(self, generator_name):
        if generator_name == 'All':
            self._train_generator = AllTripletSampler
        else:
            raise KeyError
        self.logger.info('Train Generator : ' + self._train_generator.__name__)

    def set_test_generator(self, generator_name=None):
        if generator_name is None:
            self._test_generator = TestSampler
        else:
            raise KeyError
        self.logger.info('Test Generator : ' + self._test_generator.__name__)

    def get_train(self, *args, **kwargs):
        return DataLoader(self.dataset,
                          batch_sampler=self._train_generator(self.dataset.train_info, *args, **kwargs, npr=self.npr),
                          num_workers=self.num_workers,
                          pin_memory=True)

    def get_test(self, batch_size):
        return DataLoader(self.dataset, batch_sampler=self._test_generator(self.dataset.test_info, batch_size),
                          num_workers=self.num_workers, pin_memory=True)