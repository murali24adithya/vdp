import os
import vdp
import json
import glob


def get_configs(loc="./config"):
    for (root, _, files) in os.walk(loc):
        for f in files:
            yield os.path.join(root, f)


pipeline = vdp.pipeline.Pipeline([
    vdp.convert.SGConvert(use_cache=True),
    vdp.generate.SGGenerate(use_cache=True, config=vdp.config.DEFAULT_SG_CONFIG),
    vdp.ir.SGIR(use_cache=True, **vdp.config.DEFUALT_SG_IR_CONFIG)
])


for pth in get_configs():
    print("running:", pth)
    vdp_params = vdp.utils._read_json(pth)
    config = pipeline.run(vdp_params)
