import os
import shutil
import json

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
    # del params['name']
    folder_path = os.path.join("./data/interim/", normalized_name)
    test_dir = (os.path.join(folder_path, "test"))
    train_dir = (os.path.join(folder_path, "train"))
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    #@TODO Logging
    for img_path in params['test']:
        new_img_path = (os.path.join(test_dir, os.path.basename(img_path)))
        shutil.copy(img_path, new_img_path)
    # del params['test']
    for img_path in params['train']:
        new_img_path = (os.path.join(train_dir, os.path.basename(img_path)))
        shutil.copy(img_path, new_img_path)
    # del params['train']
    
    if len(params):
        config_path = (os.path.join(folder_path, "config.json"))
        with open(config_path, 'w') as fp:
            json.dump(params, fp)  
    return normalized_name
    
def construct_fo(image_idx = 0, box_topk = 20, rel_topk = 20, sg_output_dir="./data/sg_processed/test/"):
    custom_prediction = json.load(open(f'{sg_output_dir}custom_prediction.json'))
    custom_data_info = json.load(open(f'{sg_output_dir}custom_data_info.json'))
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']
    box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
    all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
    all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
    all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

    for i in range(len(box_labels)):
        box_labels[i] = ind_to_classes[box_labels[i]]

    rel_labels = []
    rel_scores = []
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            rel_scores.append(all_rel_scores[i])
            label = (all_rel_pairs[i][0], box_labels[all_rel_pairs[i][0]], all_rel_pairs[i][1], box_labels[all_rel_pairs[i][1]],  ind_to_predicates[all_rel_labels[i]], all_rel_scores[i])
            rel_labels.append(label)

    rel_labels = rel_labels[:rel_topk]
    rel_labels = rel_labels[:rel_topk]
    constants = {lab[1] for lab in rel_labels}.union({lab[3] for lab in rel_labels})
    variables = {str(lab[0]) + "_" + lab[1] for lab in rel_labels}.union({ str(lab[2]) + "_" + lab[3] for lab in rel_labels})
    var_const_map = {**{str(lab[0]) + "_" + lab[1] : lab[1] for lab in rel_labels}, **{ str(lab[2]) + "_" + lab[3] : lab[3] for lab in rel_labels}}
    relations = dict()
    for xi, x, yi, y, r, s in rel_labels:
        if r not in relations:
            relations[r] = [(str(xi) + "_" + x, str(yi) + "_" + y)]
        else:
            relations[r].append((str(xi) + "_" + x, str(yi) + "_" + y))

    relation_signatures = relations.copy()

    for key in relation_signatures.keys():
        relation_signatures[key] = "(object, object)"

    relation_signatures['has label'] = "(object, label)"

    relations['has label'] = [list(item) for item in var_const_map.items()]
    fo_model = {
        'sorts': ['object', 'label'],
        'predicates': relation_signatures,
        'elements': {'object' : list(variables), 'label' : list(constants)},
        'interpretation': relations,
        'raw' : rel_labels
    }

    return fo_model


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