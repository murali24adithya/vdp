import os
import glob
parent_path = os.path.abspath("./../")
if os.path.basename(parent_path) == 'vdp':
    os.chdir(parent_path)
import vdp

vdp_params = vdp.utils.conf("config.json")
dry=False

vdp.utils.make_set(vdp_params)
name = vdp_params['name']
sg_params = vdp_params['sg_config']
sg_output_dir = sg_params['output_dir']
os.makedirs(sg_output_dir + f"/{name}", exist_ok=True)
os.chdir("./sg")

for batch_dir in ['train', 'test']:
        vdp.utils.run_sg(input_path=f"./../data/interim/{name}/{batch_dir}",
                output_path=f"./.{sg_output_dir}/{name}",
                glove_path=f"./.{sg_params['glove_path']}",
                model_path=f"./.{sg_params['model_path']}",
                log_path=f"./.{sg_output_dir}/{name}/run.log",
                sg_tools_rel_path=sg_params['sg_tools_rel_path'],
                sg_config_path=sg_params['sg_config_path'],
                cuda_device_port=sg_params['cuda_device_port'],
                n_proc=sg_params['n_proc'],
                dry=dry)
os.chdir("./..")
print("Done!")
