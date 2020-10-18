import os
import glob
parent_path = os.path.abspath("./../")
if os.path.basename(parent_path) == 'vdp':
    os.chdir(parent_path)
import vdp
import json

dry=True

for vdp_config_json in glob.glob("./test_config.json"):
    print("running:", vdp_config_json)
    with open(vdp_config_json) as f:
        vdp_params = json.load(f)
    vdp.utils.make_sets(vdp_params)
    name = vdp_params['name']
    test_imgs = vdp_params['test']
    train_imgs = vdp_params['train']

    if 'sg_config' in vdp_params:
        sg_params = vdp_params['sg_config']
        sg_output_dir = sg_params['output_dir']
        os.chdir("./sg")

        for batch_dir in ['train', 'test']:
            os.makedirs(sg_output_dir + f"/{name}/{batch_dir}", exist_ok=True)
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

    if 'fo_config' in vdp_params:
        fo_params = vdp_params['fo_config']
        for batch_dir in ['train', 'test']:
            sg_input_dir = fo_params['input_dir'] + f"/{name}/{batch_dir}"
            fo_output_dir = fo_params['output_dir'] + f"/{name}/{batch_dir}"
            os.makedirs(fo_output_dir, exist_ok=True)
            fo_models = vdp.utils.get_sgg_fo_models(sg_input_dir, box_topk=fo_params['box_topk'], rel_topk=fo_params['rel_topk'], raw_img_dir=os.path.dirname(test_imgs[0]))

            for img_path, fo_model in fo_models:
                output_json = f"{fo_output_dir}/{os.path.basename(img_path).replace('.jpg', '_model.json')}"
                with open(output_json, 'w') as fp:
                    json.dump(fo_model, fp)

    print("Done!")
