import os
import glob
parent_path = os.path.abspath("./../")
if os.path.basename(parent_path) == 'vdp':
    os.chdir(parent_path)
import vdp

dry=True

for vdp_config_json in glob.glob("./config/*"):
    print("running:", vdp_config_json)
    vdp_params = vdp.utils.conf(vdp_config_json)
    vdp.utils.make_set(vdp_params)
    name = vdp_params['name']

    if 'sg_config' in vdp_params:
        sg_params = vdp_params['sg_config']
        sg_output_dir = sg_params['output_dir']
        os.makedirs(sg_output_dir + f"/{name}/train", exist_ok=True)
        os.makedirs(sg_output_dir + f"/{name}/test", exist_ok=True)
        os.chdir("./sg")

        for batch_dir in ['train', 'test']:
                vdp.utils.run_sg(input_path=f"./../data/interim/{name}/{batch_dir}",
                        output_path=f"{sg_output_dir}/{name}/{batch_dir}",
                        glove_path=f"{sg_params['glove_path']}",
                        model_path=f"{sg_params['model_path']}",
                        log_path=f"{sg_output_dir}/{name}/{batch_dir}/run.log",
                        sg_tools_rel_path=sg_params['sg_tools_rel_path'],
                        sg_config_path=sg_params['sg_config_path'],
                        cuda_device_port=sg_params['cuda_device_port'],
                        n_proc=sg_params['n_proc'],
                        dry=dry)
        os.chdir("./..")

    if 'fo_config' in vdp_params and not dry:
        fo_params = vdp_params['fo_config']
        for batch_dir in ['train', 'test']:
            sg_input_dir = fo_params['input_dir'] + f"/{name}/{batch_dir}"
            fo_output_dir = fo_params['output_dir'] + f"/{name}/{batch_dir}"
            os.makedirs(fo_output_dir, exist_ok=True)
            vdp.utils.construct_fo(sg_input_dir, fo_output_dir, box_topk=fo_params['box_topk'], rel_topk=fo_params['rel_topk'])

    print("Done!")
