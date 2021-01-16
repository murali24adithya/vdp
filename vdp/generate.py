from .convert import *
from .utils import _read_pickle, _read_json, _to_pickle
from .config import InputConfig, DEFAULT_SG_CONFIG
from .pipeline import Pipe
import os


class SGGenerate(Pipe):
    def __init__(self, config = DEFAULT_SG_CONFIG, use_cache=True):
        Pipe.__init__(self, use_cache=use_cache)
        self.sg_config = config
        self.use_cache = use_cache
        self.cache = _read_pickle("./data/cache.pkl") if (use_cache and os.path.exists("./data/cache.pkl")) else dict()
            
    def run_sg(self, input_path, output_path, glove_path, model_path, log_path, sg_tools_rel_path="tools/relation_test_net.py", sg_config_path="configs/e2e_relation_X_101_32_8_FPN_1x.yaml", cuda_device_port=0, n_proc=1, dry=True):
        """
        Inputs: 
        input_path: str
            The location of the directory with input images.
            This folder must not contain anything other than the images.
        output_path: str
            The location of the output directory.
            This folder must be empty.
        glove_path: str
            The location of the word embeddings.
            If folder is empty, word embeddings will be downloaded to location
        model_path: str
            The location of the trained scene graph generator.
        log_path: str
            The location where the log file should be written to.
        sg_tools_rel_path: str
            The location of the scene graph controller.
        cuda_device_port: int
            The port of the GPU
        n_proc: int
            Number of processes scene graph controller should spawn.
        Notes:
        """
        # the input paths are all relative to the base directory. need to change that.
        pth = "./sg"
        input_path = os.path.relpath(input_path, pth)
        output_path = os.path.relpath(output_path, pth)
        glove_path = os.path.relpath(glove_path, pth)
        model_path = os.path.relpath(model_path, pth)
        log_path = os.path.relpath(log_path, pth)
        sg_tools_rel_path = os.path.relpath(sg_tools_rel_path, pth)
        sg_config_path = os.path.relpath(sg_config_path, pth)

        os.chdir("./sg")

        cmd = f"""CUDA_VISIBLE_DEVICES={cuda_device_port}
        python -m torch.distributed.launch --master_port 10027
        --nproc_per_node={n_proc}
        {sg_tools_rel_path}
        --config-file "{sg_config_path}"
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
        MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
        MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE
        MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum 
        MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs 
        TEST.IMS_PER_BATCH 1 
        DTYPE "float16" 
        GLOVE_DIR {glove_path}
        MODEL.PRETRAINED_DETECTOR_CKPT {model_path}
        OUTPUT_DIR {model_path} 
        TEST.CUSTUM_EVAL True
        TEST.CUSTUM_PATH {input_path}
        DETECTED_SGG_DIR {output_path}
        > {log_path}
        """.replace("\n", " ").replace("    ", "")
        if dry:
            print("DRY RUN: ", cmd)
        else:
            os.system(cmd)
        os.chdir("./..")

    def __call__(self, params):
        super().__call__(self)
        self.config = params
        sg_processed_dir = os.path.join(self.sg_config.output_dir, os.path.basename(self.config.interim_path))
        log_path = os.path.join(sg_processed_dir, "run.log")
        os.makedirs(sg_processed_dir, exist_ok=True)

        # (self.config.interim_path)
        self.run_sg(input_path=self.config.interim_path,
            output_path = sg_processed_dir,
            glove_path=self.sg_config.glove_path, 
            model_path=self.sg_config.model_path,
            log_path=log_path,
            sg_tools_rel_path=self.sg_config.sg_tools_rel_path,
            sg_config_path=self.sg_config.sg_config_path,
            cuda_device_port=self.sg_config.cuda_device_port,
            n_proc=self.sg_config.n_proc, 
            dry=self.config.dry)
        
        self.config.sg_processed_dir = sg_processed_dir            

        return self.config


class YOLOGenerate(Pipe):
    # Nothing happens here.
    # @TODO use YOLOv3.
    def __init__(self, use_cache=True):
        Pipe.__init__(self, use_cache=use_cache)
        self.use_cache = use_cache
        self.cache = _read_pickle("./data/cache.pkl") if (use_cache and os.path.exists("./data/cache.pkl")) else dict()
            

    def __call__(self, params):
        super().__call__(self)
        self.config = params
        return self.config



if __name__ == '__main__':
    obj = YOLOConvert(use_cache=True)
    # params = obj.__call__(_read_json("./config/test.json"))
    # obj.__save__()
    # obj2 = YOLOGenerate(config=DEFAULT_SG_CONFIG, use_cache=True)
    # obj2.__call__(params)

    # obj2.__save__()
