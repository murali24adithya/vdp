import os
from datetime import datetime
from .utils import _read_pickle, _read_json, _to_pickle, _to_json, _read_txt
from .pipeline import Pipe
from .config import *
from .convert import *
from .generate import *


class SGIR(Pipe):
    def __init__(self, output_path, use_cache=True):
        Pipe.__init__(self, use_cache=use_cache)
        self.output_path = output_path
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
            fo_models = self.get_sgg_fo_models(self.config.processed_path, self.config.raw_path)

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
            output_dir = os.path.join(self.output_path, self.config.name, partition)
            os.makedirs(output_dir, exist_ok=True)
            _to_json(model, os.path.join(output_dir, os.path.basename(img)) + ".json")

        all_images = map(lambda pth: os.path.basename(pth), self.config.train + self.config.test)
        remaining_images = list(set(images_processed) - set(all_images)) if self.use_cache else []

        for img in remaining_images:
            partition = 'train' if os.path.basename(img) in self.config.train else 'test'
            output_dir = os.path.join(self.output_path, self.config.name, partition)
            os.makedirs(output_dir, exist_ok=True)
            _to_json(self.cache[img]['sg_ir'] , os.path.join(output_dir, os.path.basename(img)) + ".json")

        print(f"IR outputs written to @ `{os.path.dirname(output_dir)}`")

        if self.use_cache: 
            _to_pickle(self.cache, "./data/cache.pkl")
        else:
            os.makedirs("./data", exist_ok=True)
            _to_pickle(self.cache, "./data/temp.pkl")





class YOLOIR(Pipe):
    def __init__(self, output_path, use_cache=True):
        Pipe.__init__(self, use_cache=use_cache)
        self.output_path = output_path
        self.use_cache = use_cache
        self.cache = _read_pickle("./data/cache.pkl") if (use_cache and os.path.exists("./data/cache.pkl")) else dict()
        self.idx2coco = {str(i) : c for i, c in enumerate(_read_txt("./darknet/data/coco.names"))}


    def dist(self, c1, c2):
        """Measure euclidean distance between centroids"""
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
    def frame(self, rc):
        """Return the left, up, right, down `bounds` given the center, width, and height of img"""
        left  = rc['center_x'] - 0.5 * rc['width']
        up    = rc['center_y'] + 0.5*  rc['height']
        right = rc['center_x'] + 0.5*  rc['width']
        down  = rc['center_y'] - 0.5 * rc['height']
        return (left, up, right, down) # Following YOLO syntax

    def relate(self, c1, c2):
        """Relate position of c1 w.r.t c2. `c1 is to the ___ of c2`"""
        x_rel = "right" if c1[0] >= c2[0] else "left"
        y_rel = "above" if c1[1] >= c2[1] else "below"
        return (x_rel, y_rel)
        
    def is_within(self, bb1, bb2):
        """Whether bb1 is within bb2."""
        return (bb1[0] >= bb2[0]) and (bb1[2] <= bb2[2]) and (bb1[1] >= bb2[1]) and (bb1[3] <= bb2[3])


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

    def _clean_image_paths(self, img_list, raw_directory):
        # A helped function that corrects the image locations 
        # stored by the scene graph generator.
        path_cleaner = lambda pth: os.path.join(raw_directory, os.path.basename(pth))
        return list(map(path_cleaner, img_list))

    def get_yolo_fo_models(self, yolo_input_dir):
        pred_path = os.path.join(yolo_input_dir, 'predictions.json')
        preds = _read_json(pred_path)
        fo_models = list()
        for img_pred in preds:
            rel_labels = list()
            var_const_map = dict()
            objs = img_pred['objects']
            object_id = {obj['name'] : str(i) for i, obj in enumerate(objs)}
            for obj1 in objs:
                for obj2 in objs:
                    if obj1 != obj2:
                        rc1, rc2 = obj1['relative_coordinates'], obj2['relative_coordinates']
                        c1 = rc1['center_x'], rc1['center_y']
                        c2 = rc2['center_x'], rc2['center_y']
                        conf = obj1['confidence'] * obj2['confidence']
                        relations = self.relate(c1, c2)
                        if self.is_within(self.frame(rc1), self.frame(rc2)):
                            relations.append('within')
                        for rel in relations:
                            rel_labels.append([object_id[obj1['name']], obj1['name'], object_id[obj2['name']], obj2['name'], rel, str(conf)])

                var_const_map[object_id[obj1['name']] + "_" + obj1['name']] = obj1['name']

            fo_model = self._construct_normalized_model(rel_labels, var_const_map)
            img_path =  os.path.relpath(img_pred['filename'], ".")
            fo_models.append((img_path, fo_model))
            
        return fo_models
    

    def __call__(self, params):
        super().__call__(self)
        self.config = params
        fo_models = list()
        if not self.config.dry:
            fo_models = self.get_yolo_fo_models(self.config.processed_path)

        if self.use_cache and (not self.config.dry):
            for (pth, model) in fo_models:
                    img = os.path.basename(pth)
                    self.cache[img]['yolo_ir'] = model

        self.config.fo_models = fo_models

        return self.config

    def __save__(self):
        images_processed = list()
        for (pth, model) in self.config.fo_models:
            img = os.path.basename(pth)
            images_processed.append(img)
            partition = 'train' if img in list(map(os.path.basename, self.config.train)) else 'test'
            output_dir = os.path.join(self.output_path, self.config.name, partition)
            os.makedirs(output_dir, exist_ok=True)
            _to_json(model, os.path.join(output_dir, img) + ".json")

        all_images = list(map(os.path.basename, self.config.train + self.config.test))
        remaining_images = list(set(all_images) - set(images_processed)) if self.use_cache else []

        for pth in remaining_images:
            img = os.path.basename(pth)
            partition = 'train' if img in list(map(os.path.basename, self.config.train)) else 'test'
            output_dir = os.path.join(self.output_path, self.config.name, partition)
            os.makedirs(output_dir, exist_ok=True)
            _to_json(self.cache[img]['yolo_ir'] , os.path.join(output_dir, os.path.basename(img)) + ".json")

        print(f"IR outputs written to @ `{self.output_path}`")

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