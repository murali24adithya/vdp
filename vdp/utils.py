import os
import shutil
import json

# os.chdir(os.path.abspath("./../")) # point this to the project directory 


def conf(config_path):
    with open(config_path) as f:
        return json.load(f)
    
def make_set(params):
    """
    Inputs: 
    vdp_params: dict
        A json file containing formatted dict
    
    Notes:
        dict must be formatted as such:
        params = {
            "name"  : "str", 
            "train" : ["1.jpg", "2.jpg", "3.jpg"],
            "test"  : ["4.jpg", "5.jpg", "6.jpg"]
        }
    """
    #@TODO Input verification
    normalized_name = params["name"].replace(" ", "_").lower()
    del params['name']
    folder_path = os.path.join("./data/interim/", normalized_name)
    test_dir = (os.path.join(folder_path, "test"))
    train_dir = (os.path.join(folder_path, "train"))
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    #@TODO Logging
    for img_path in params['test']:
        new_img_path = (os.path.join(test_dir, os.path.basename(img_path)))
        shutil.copy(img_path, new_img_path)
    del params['test']

    for img_path in params['train']:
        new_img_path = (os.path.join(train_dir, os.path.basename(img_path)))
        shutil.copy(img_path, new_img_path)

    del params['train']
    
    if len(params):
        config_path = (os.path.join(folder_path, "config.json"))
        with open(config_path, 'w') as fp:
            json.dump(params, fp)  
    return normalized_name
    

def run_sg(input_path, output_path, glove_path, model_path, log_path, sg_tools_rel_path="tools/relation_test_net.py", sg_config_path="configs/e2e_relation_X_101_32_8_FPN_1x.yaml", cuda_device_port=0, n_proc=1, dry=True):
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
    print(cmd)
    if not dry:
        os.system(cmd)

def controller(file, dry=True):
    vdp_params = conf(file)
    sg_params = vdp_params['sg_config']
    name = make_set(vdp_params)
    sg_output_dir = sg_params['output_dir']
    os.makedirs(sg_output_dir + f"/{name}", exist_ok=True)
    run_sg(input_path=f"./data/interim/{name}/test",
           output_path=f"{sg_output_dir}/{name}",
           glove_path=sg_params['glove_path'],
           model_path=sg_params["model_path"],
           log_path=f"{sg_output_dir}/{name}/run.log",
           sg_tools_rel_path=sg_params['sg_tools_rel_path'],
           sg_config_path=sg_params['sg_config_path'],
           cuda_device_port=sg_params['cuda_device_port'],
           n_proc=sg_params['n_proc'],
           dry=dry)
    print("Done!")
