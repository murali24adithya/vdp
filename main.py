import os
import vdp
import argparse
main_parser = argparse.ArgumentParser()
main_parser.add_argument("-s", "--use_sg", action="store_true", help="If enabled, the SG detector is run instead of the YOLO detector.", default=False)
main_parser.add_argument("-d", "--dry", action="store_true", help="If enabled, the NN engine is not run. Usually for debugging purposes.", default=False)
main_parser.add_argument("-c", "--no_cache", action="store_true", help="If enabled, cached images/results are not used.", default=False)
args = main_parser.parse_args()
print(args)

use_cache = not args.no_cache


def get_configs(loc="./config"):
    """
    Returns a generator with the path of config json's in `{project dir}/config`
    Only used for "batch" processing.
    """
    for (root, _, files) in os.walk(loc):
        for f in files:
            yield os.path.join(root, f)

if args.use_sg:
    pipeline = vdp.pipeline.Pipeline([
        vdp.convert.SGConvert(use_cache=use_cache),
        vdp.generate.SGGenerate(use_cache=use_cache, config=vdp.config.DEFAULT_SG_CONFIG),
        vdp.ir.SGIR(use_cache=use_cache, **vdp.config.DEFAULT_SG_IR_CONFIG)
    ])
else:
    pipeline = vdp.pipeline.Pipeline([
        vdp.convert.YOLOConvert(use_cache=use_cache),
        vdp.generate.YOLOGenerate(use_cache=use_cache, config=vdp.config.DEFAULT_YOLO_CONFIG),
        vdp.ir.YOLOIR(use_cache=use_cache, **vdp.config.DEFAULT_YOLO_IR_CONFIG)
    ])

for pth in get_configs():
    print("running:", pth)
    vdp_params = vdp.utils._read_json(pth)
    config = pipeline.run(vdp_params)