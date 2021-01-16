
from dataclasses import dataclass
from collections import namedtuple

SGConfig = namedtuple("SGConfig", ['output_dir', 'glove_path', 'model_path', 'sg_tools_rel_path', 'sg_config_path', 'cuda_device_port', 'n_proc'])
FOConfig = namedtuple("FOConfig", ['raw_img_dir', 'output_dir'])


DEFAULT_SG_CONFIG = SGConfig(**{
    "output_dir": "./data/sg_processed",
    "glove_path": "./../checkpoints/glove", 
    "model_path": "./../checkpoints/model", 
    "sg_tools_rel_path": "./sg/tools/relation_test_net.py", 
    "sg_config_path": "./sg/configs/e2e_relation_X_101_32_8_FPN_1x.yaml", 
    "cuda_device_port": 0, 
    "n_proc": 1
    })


DEFAULT_YOLO_CONFIG = SGConfig(**{
    "output_dir": "./data/yolo_processed",
    "glove_path": "./../checkpoints/glove", 
    "model_path": "./../checkpoints/model", 
    "sg_tools_rel_path": "./sg/tools/relation_test_net.py", 
    "sg_config_path": "./sg/configs/e2e_relation_X_101_32_8_FPN_1x.yaml", 
    "cuda_device_port": 0, 
    "n_proc": 1
    })


DEFUALT_IR_CONFIG = FOConfig(**{
    "output_dir": "./data/sg_ir",
    "raw_img_dir": "./data/raw"
    })


@dataclass
class InputConfig:
    """Class for keeping track of pipeline variables."""
    name: str
    train: list
    test: list
    interim_path: str = None
    data: str = None
    dry: str = None
    yolo_path: str = None
    sg_processed_dir: str = None
    fo_models: str = None
