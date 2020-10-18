import os
import shutil
import json
import re
rx_dict = {
    'start' : re.compile(r"Enter Image Path: .+\/(?P<name>[^/]+).jpg: Predicted in \d*\.?\d* seconds.\n"),
    'obj' : re.compile(r"(?P<obj>\w+): (?P<score>\d+)%\n"),
    'bb' : re.compile(r"Bounding Box: Left=(?P<left>\d+), Top=(?P<top>\d+), Right=(?P<right>\d+), Bottom=(?P<bottom>\d+)")
}


def _parse_line(line):
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return (key, match)
    return (None, None)


def _parse_file(file_path):
    data = list()
    
    with open(file_path, 'r') as fp:
        line = fp.readline()
        while 'Predicted' in line:
            key, match = _parse_line(line)
            name = match.group('name')
            img_line = fp.readline()
            img_data = dict()
            objs = list()
            while img_line and 'Predicted' not in img_line:       
                key, match = _parse_line(img_line)
                if key == 'obj':
                    objs.append((match.group('obj'), float(match.group('score'))))

                if key == 'bb':
                    bb = (match.group('left'), match.group('top'), match.group('bottom'), match.group('right'))
                    img_data = {**img_data, **{obj : {'score' : score, 'bb' : bb} for obj, score in objs}}
                    objs = list()
                
                img_line = fp.readline()
            
            data.append((name, img_data))
            line = img_line
    return data
    
def make_sets(params):
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
    


def _construct_normalized_model(rel_labels):
    constants = {lab[1] for lab in rel_labels}.union({lab[3] for lab in rel_labels})
    scores = [lab[5] for lab in rel_labels]
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
    relation_signatures['has score'] = "(object, object, score)"

    relations['has label'] = [tuple(item) for item in var_const_map.items()]
    relations['has score'] = [(str(xi) + "_" + x, str(yi) + "_" + y, s)  for xi, x, yi, y, r, s in rel_labels]
    fo_model = {
        'sorts': ['object', 'label', 'scores'],
        'predicates': relation_signatures,
        'elements': {'object' : list(variables), 'label' : list(constants), 'scores' : list(scores)},
        'interpretation': relations,
        'raw' : rel_labels
    }
    return fo_model

def _clean_image_paths(img_list, raw_directory="./data/raw/"):
    """
    A helped function that corrects the image locations stored by the scene graph generator.

    Inputs:
        img_list: list
            This is the image list stored inside custom_data_info['idx_to_files']
        raw_directory: str
            The folder contaning the 'correct' image

    Outputs:
        anon: list
            corrected list of image paths.
    """
    return [raw_directory + os.path.basename(img_path) for img_path in img_list]

def get_sgg_fo_models(sg_input_dir, box_topk = 20, rel_topk = 20, raw_img_dir = "./data/raw/"):
    custom_prediction = json.load(open(f'{sg_input_dir}/custom_prediction.json'))
    custom_data_info = json.load(open(f'{sg_input_dir}/custom_data_info.json'))
    image_paths = _clean_image_paths(custom_data_info['idx_to_files'], raw_directory=raw_img_dir)
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']

    fo_models = list()
    for image_idx in custom_prediction.keys():
        img_path = image_paths[int(image_idx)]
        box_labels = custom_prediction[str(image_idx)]['bbox_labels'][:box_topk]
        all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
        all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
        all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']

        for i in range(len(box_labels)):
            box_labels[i] = ind_to_classes[box_labels[i]]

        rel_labels = []
        for i in range(len(all_rel_pairs)):
            if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
                label = (all_rel_pairs[i][0], box_labels[all_rel_pairs[i][0]], all_rel_pairs[i][1], box_labels[all_rel_pairs[i][1]],  ind_to_predicates[all_rel_labels[i]], all_rel_scores[i])
                rel_labels.append(label)

        rel_labels = rel_labels[:rel_topk]

        fo_model = _construct_normalized_model(rel_labels)
        fo_models.append((img_path, fo_model))

    return fo_models

def get_yolo_fo_models(yolo_input_dir):
    print(yolo_input_dir)
    fo_models = list()
    dir_prefix = yolo_input_dir + f"/{os.path.basename(yolo_input_dir)}"
    train_path = dir_prefix + "_train_out.txt"
    test_path = dir_prefix + "_test_out.txt"
    data = [*_parse_file(train_path), *_parse_file(test_path)]
    
    centroid = lambda bb : ((int(bb[0]) + int(bb[2])) / 2, (int(bb[1]) + int(bb[3])) / 2)
    # dist = lambda c1, c2 : ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
    relate = lambda c1, c2 : ["right" if c1[0] >= c2[0] else "left", "above" if c1[1] >= c2[1] else "below"]

    for img_name, img_data in data:
        batch = img_name.split("_")[0]
        os.makedirs(f"{dir_prefix}/{batch}", exist_ok=True)
        rel_labels = list()
        for obj1, obj_data1 in img_data.items():
            for obj2, obj_data2 in img_data.items():
                if obj1 != obj2:
                    c1 = centroid(obj_data1['bb'])
                    c2 = centroid(obj_data2['bb'])
                    rel_labels.append(['obj', obj1, 'obj', obj2, "_".join(relate(c1, c2)), obj_data1['score'] * obj_data2['score'] * 1e-4])

        fo_model = _construct_normalized_model(rel_labels)
        img_path =  yolo_input_dir + f"/{img_name}.jpg"
        fo_models.append((img_path, fo_model))
    
    return fo_models

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