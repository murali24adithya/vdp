from dataclasses import dataclass, asdict
from collections import namedtuple

@dataclass
class Config:
    """Class for keeping track of pipeline variables."""
    name: str                                   # name of puzzle
    train: list                                 # image paths belonging to 'train' set
    test: list                                  # image paths belonging to 'test' set
    processed_path: str = None                  # path of nn output. 
    raw_path: str = "./data/images"             # path of raw folder. Defaults to "{project dir}/data/images"
    interim_path: str = "./data/interim"        # path of collated results. Defaults to "{project dir}/data/interim"
    dry: str = None                             # Whether to run the (conputationally expensive) neural layer (sg/yolo)
    fo_models: dict = None                      # The first order models generated.

@dataclass
class SceneGraphConfig:
    """Class for keeping track of variables needed by scene graph detector"""
    glove_path: str = './../checkpoints/glove'                                      # Path of GLOVE word vectors. defaults to {project dir}/../checkpoints/glove
    model_path: str = './../checkpoints/model'                                      # Path of pretrained model. If folder is empty, word embeddings will be downloaded to location. defaults to {project dir}/../checkpoints/model
    launch_script_path: str =    "./sg/tools/relation_test_net.py"                  # Path to `relation_test_net.py`. defaults to {project dir}/sg/
    maskrcnn_config_path: str = "./sg/configs/e2e_relation_X_101_32_8_FPN_1x.yaml"  # Path of MASKRCNN config yaml. defaults to 
    cuda_device_port: str = "0"                                                     # CUDA port of GPU(s)
    n_proc: int = 1                                                                 # Number of processes to spawn.

@dataclass
class YOLOConfig:
    """Class for keeping track of variables needed by darknet"""
    model_path: str = './darknet/yolov4.weights'                                      # Path of pretrained model. defaults to {project dir}/darknet/yolov4.weights
    output_dir: str = './data/yolo_processed'                                      # Path of pretrained model. defaults to {project dir}/darknet/yolov4.weights


@dataclass
class FOConfig:
    """Class for keeping track of variables needed by darknet"""
    output_dir: str = './data/ir'
    raw_img_dir: str = './darknet/yolov4.weights'                                      # Path of pretrained model. defaults to {project dir}/darknet/yolov4.weights

SGConfig = namedtuple("SGConfig", ['output_dir', 'glove_path', 'model_path', 'sg_tools_rel_path', 'sg_config_path', 'cuda_device_port', 'n_proc'])
FOConfig = namedtuple("FOConfig", ['raw_img_dir', 'output_dir'])

DEFAULT_SG_CONFIG = SceneGraphConfig()
DEFAULT_YOLO_CONFIG = YOLOConfig()
DEFUALT_YOLO_IR_CONFIG = dict(output_path='./data/yolo_ir')
DEFAULT_SG_IR_CONFIG = dict(output_path='./data/sg_ir')


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
