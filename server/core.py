import sys
import getopt

host_ip = '0.0.0.0'
host_port = 2000
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "default"
device = "cuda"

for op, value in getopt.getopt(sys.argv[1:], "hc:t:d:i:p:")[0]:
    if op == '-h':
        print(
            "SAM_Flask Server\nUsage: python3 main.py [-h] [-c checkpoint_path] [-t model_type] [-d device]\n")
        print("Options:")
        print(f"  -h                 : this help")
        print(f"  -i host ip         : set host ip (default: 0.0.0.0)")
        print(f"  -p port            : set server port (default: 2000)")
        print(f"  -d device          : set device to run SAM (default: cuda)")
        print(f"  -t model type      : set type of model (default: vit_h)")
        print(
            f"  -c checkpoint path : set path to checkpoint (default: sam_vit_h_4b8939.pth)")
        sys.exit()
    elif op == '-c':
        sam_checkpoint = value
    elif op == '-t':
        model_type = value
    elif op == '-d':
        device = value
    elif op == '-i':
        host_ip = value
    elif op == '-p':
        host_port = value


from flask import *
from segment_anything import sam_model_registry, SamPredictor


class SAM_Flask(Flask):
    def __init__(self, sam_checkpoint, model_type, device, import_name='SAM_Flask', static_url_path=None, static_folder="static", static_host=None, host_matching=False, subdomain_matching=False, template_folder="templates", instance_path=None, instance_relative_config=False, root_path=None):
        super().__init__(import_name, static_url_path, static_folder, static_host, host_matching,
                         subdomain_matching, template_folder, instance_path, instance_relative_config, root_path)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)


Flask_app = SAM_Flask(sam_checkpoint=sam_checkpoint,
                      model_type=model_type, device=device)
