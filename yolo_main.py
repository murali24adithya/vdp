import os
import vdp


def get_configs(loc="./config"):
    for (root, _, files) in os.walk(loc):
        for f in files:
            yield os.path.join(root, f)


pipeline = vdp.pipeline.Pipeline([
    vdp.convert.YOLOConvert(use_cache=True),
    vdp.generate.YOLOGenerate(use_cache=True, config=vdp.config.DEFAULT_YOLO_CONFIG),
    vdp.ir.YOLOIR(use_cache=True, **vdp.config.DEFUALT_YOLO_IR_CONFIG)
])


for pth in get_configs():
    print("running:", pth)
    vdp_params = vdp.utils._read_json(pth)
    config = pipeline.run(vdp_params)
