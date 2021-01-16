import os
import shutil
import json
import re
import pickle
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


def _read_json(path):
    with open(path, 'r') as fh:
        return json.load(fh)

def _to_json(obj, path):
    with open(path, 'w') as fh:
        return json.dump(obj, fh)

def _read_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def _to_pickle(obj, path):
    with open(path, 'wb') as fh:
        return pickle.dump(obj, fh, protocol=4)