import os
import shutil
from datetime import datetime
from .utils import _read_pickle, _read_json, _to_pickle
from .config import InputConfig
from .pipeline import Pipe


class SGConvert(Pipe):
    def __init__(self, use_cache=True):
        Pipe.__init__(self, use_cache=use_cache)

    def filter_images(self, config):
        for pth in config.train + config.test:
            img = os.path.basename(pth)
            # process an image if either we are disregarding the cache
            # or if we are using the cache and the image isn't part of it
            to_process = (not self.use_cache) or (self.use_cache and img not in self.cache)
            if to_process:
                self.cache[os.path.basename(img)] = {
                    'sg_ir': dict(),
                    'yolo_ir': dict(),
                    'sg_proc': dict(),
                    'yolo_proc' : dict()
                }
                yield pth


    def make_sg_set(self, imgs_to_run):
        # make an interim folder to "process" the images
        # This is required by the NN engine.
        name = "run_on_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
        interim_path =  os.path.join("./data/interim", name) 
        os.makedirs(interim_path, exist_ok=True)
        # copy files over to the interim folder.
        for img_path in imgs_to_run:
            assert(os.path.exists(img_path)), f"Couldn't find `{img_path}` in `{self.config.name}`"
            new_path = os.path.join(interim_path, os.path.basename(img_path))
            shutil.copy(img_path, new_path)
        return interim_path


    def __call__(self, params):
        # transfer all the "new" images into an interim folder.
        super().__call__(self)
        self.config = InputConfig(**params)
        imgs_to_run = self.filter_images(self.config)
        interim_path = self.make_sg_set(imgs_to_run)
        self.config.dry = len(os.listdir(interim_path)) == 0
        self.config.interim_path = interim_path
        return self.config



# @TODO.
# class YOLOConvert(Pipe):
    # def filter_images(self, config):
    #     for img in config.train + config.test:
    #         # process an image if either we are disregarding the cache
    #         # or if we are using the cache and the image isn't part of it
    #         to_process = (not self.use_cache) or (self.use_cache and img not in self.cache)
    #         if to_process:
    #             self.cache[os.path.basename(img)] = {
    #                 # 'sg_ir': dict(),
    #                 # 'yolo_ir': dict(),
    #                 'sg_proc': dict(),
    #                 'yolo_proc' : dict()}
    #             yield img

    # def make_sg_set(self, imgs_to_run):
    #     # make an interim folder to "process" the images
    #     # This is required by the NN engine.
    #     name = "run_on_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
    #     interim_path =  os.path.join("./data/interim", name) 
    #     os.makedirs(interim_path, exist_ok=True)
    #     # copy files over to the interim folder.
    #     for img_path in imgs_to_run:
    #         assert(os.path.exists(img_path)), f"Couldn't find `{img_path}` in `{self.config.name}`"
    #         new_path = os.path.join(interim_path, os.path.basename(img_path))
    #         shutil.copy(img_path, new_path)
    #     return interim_path


    # def __call__(self, params):
    #     # transfer all the "new" images into an interim folder.
    #     self.config = InputConfig(**params)
    #     imgs_to_run = self.filter_images(self.config)
    #     interim_path = self.make_sg_set(imgs_to_run)
    #     self.config.interim_path = interim_path
    #     return self.config

        

        
        

if __name__ == '__main__':
    obj = SGConvert(use_cache=True)
    obj.__call__(_read_json("./config/test.json"))
    obj.__save__()
