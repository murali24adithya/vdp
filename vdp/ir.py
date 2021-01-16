import os
from datetime import datetime
from .utils import _read_pickle, _read_json, _to_pickle, _to_json
from .pipeline import Pipe
from .config import *
from .convert import *
from .generate import *


class SGIR(Pipe):
    def __init__(self, config = DEFUALT_IR_CONFIG, use_cache=True):
        Pipe.__init__(self, use_cache=use_cache)
        self.ir_config = config
        self.use_cache = use_cache
        self.cache = _read_pickle("./data/cache.pkl") if (use_cache and os.path.exists("./data/cache.pkl")) else dict()


    def _construct_normalized_model(self, rel_labels, var_const_map=None):
        if not var_const_map:
            var_const_map = {**{str(lab[0]) + "_" + lab[1] : lab[1] for lab in rel_labels}, **{ str(lab[2]) + "_" + lab[3] : lab[3] for lab in rel_labels}}
        constants = set({lab for lab in var_const_map.values()})
        scores = [lab[5] for lab in rel_labels]
        rel_scores = [(str(xi) + "_" + x, str(yi) + "_" + y, s)  for xi, x, yi, y, r, s in rel_labels]
        variables = set(var_const_map.keys())
        relations = dict()
        for xi, x, yi, y, r, s in rel_labels:
            if r not in relations:
                relations[r] = [(str(xi) + "_" + x, str(yi) + "_" + y)]
            else:
                relations[r].append((str(xi) + "_" + x, str(yi) + "_" + y))

        relation_signatures = relations.copy()

        for key in relation_signatures.keys():
            relation_signatures[key] = ['object', 'object']

        relation_signatures['has_label'] = ['object', 'label']
        relations['has_label'] = [tuple(item) for item in var_const_map.items()]

        fo_model = {
            'sorts': ['object', 'label'],
            'predicates': relation_signatures,
            'elements': {'object' : list(variables), 'label' : list(constants)},
            'interpretation': relations,
            'raw' : {
                'rel_labels' : rel_labels,
                'var_const_map' : var_const_map,
                'scores' : rel_scores,
            }
        }
        return fo_model

    def _clean_image_paths(self, img_list, raw_directory="./data/raw/"):
        # A helped function that corrects the image locations 
        # stored by the scene graph generator.
        path_cleaner = lambda pth: os.path.join(raw_directory, os.path.basename(pth))
        return list(map(path_cleaner, img_list))

    def get_sgg_fo_models(self, sg_input_dir, raw_img_dir):
        pred_path = os.path.join(sg_input_dir, 'custom_prediction.json')
        data_path = os.path.join(sg_input_dir, 'custom_data_info.json')

        custom_prediction = _read_json(pred_path)
        custom_data_info = _read_json(data_path)
        ind_to_classes = custom_data_info['ind_to_classes']
        ind_to_predicates = custom_data_info['ind_to_predicates']
        image_paths = self._clean_image_paths(custom_data_info['idx_to_files'], raw_directory=raw_img_dir)

        fo_models = list()
        for image_idx in custom_prediction.keys():
            img_path = image_paths[int(image_idx)]
            all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
            all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
            all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']
            box_labels = list(map(lambda l: ind_to_classes[l], custom_prediction[str(image_idx)]['bbox_labels']))
            rel_labels = list()
            for i in range(len(all_rel_pairs)):
                idx1, idx2 = all_rel_pairs[i]
                label = (idx1, box_labels[idx1], idx2, box_labels[idx2], ind_to_predicates[all_rel_labels[i]], str(all_rel_scores[i]))
                rel_labels.append(label)

            fo_model = self._construct_normalized_model(rel_labels)
            fo_models.append((os.path.basename(img_path), fo_model))
        return fo_models
    

    def __call__(self, params):
        super().__call__(self)
        self.config = params
        if not self.config.dry:
            fo_models = self.get_sgg_fo_models(self.config.sg_processed_dir, self.ir_config.raw_img_dir)

        self.config.fo_models = list()
        if self.use_cache and (not self.config.dry):
            for (img, model) in fo_models:
                    self.cache[img]['sg_ir'] = model

        self.config.fo_models = fo_models

        return self.config

    def __save__(self):
        images_processed = list()
        for (img, model) in self.config.fo_models:
            images_processed.append(img)
            partition = 'train' if os.path.basename(img) in self.config.train else 'test'
            output_dir = os.path.join(self.ir_config.output_dir, self.config.name, partition)
            os.makedirs(output_dir, exist_ok=True)
            _to_json(model, os.path.join(output_dir, os.path.basename(img)) + ".json")

        all_images = map(lambda pth: os.path.basename(pth), self.config.train + self.config.test)
        remaining_images = list(set(images_processed) - set(all_images)) if self.use_cache else []

        for img in remaining_images:
            partition = 'train' if os.path.basename(img) in self.config.train else 'test'
            output_dir = os.path.join(self.ir_config.output_dir, self.config.name, partition)
            os.makedirs(output_dir, exist_ok=True)
            _to_json(self.cache[img]['sg_ir'] , os.path.join(output_dir, os.path.basename(img)) + ".json")

        print(f"IR outputs written to @ `{os.path.dirname(output_dir)}`")

        if self.use_cache: 
            _to_pickle(self.cache, "./data/cache.pkl")
        else:
            os.makedirs("./data", exist_ok=True)
            _to_pickle(self.cache, "./data/temp.pkl")




if __name__ == "__main__":
    obj = SGConvert(use_cache=True)
    params = obj.__call__(_read_json("./config/2on1_ace-d123.json"))
    obj.__save__()
    obj2 = SGGenerate(config=DEFAULT_SG_CONFIG, use_cache=True)
    params2 = obj2.__call__(params)
    obj2.__save__()
    obj3 = SGIR(config=DEFUALT_IR_CONFIG, use_cache=True)
    obj3.__call__(params2)
    obj3.__save__()