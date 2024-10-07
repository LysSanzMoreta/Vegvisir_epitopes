#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""


import os,sys
from gooey import Gooey, GooeyParser
from argparse import RawTextHelpFormatter
local_repository=True

# if getattr(sys, 'frozen', False):
#     script_dir = os.path.dirname(sys.executable)
# elif __file__:
#     script_dir = os.path.dirname(__file__)

# if local_repository: #TODO: The local imports are extremely slow
#      print("Importing local repository")
#      sys.path.insert(1, "{}/vegvisir/src".format(script_dir))
#      import vegvisir
# else:#pip installed module
#      import vegvisir


if getattr(sys, 'frozen', False):
    # If running in a PyInstaller bundle
    script_dir = sys._MEIPASS
else:
    # Running in normal Python environment
    script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(script_dir, 'vegvisir/src'))


import vegvisir

import Vegvisir_analysis as VegvisirAnalysis

import Vegvisir_example as VegvisirExample

if "CUDA_VISIBLE_DEVICES" in os.environ:
    device = "cuda:{}".format(os.environ['CUDA_VISIBLE_DEVICES'])
else:
    print("Cuda device has not been specified in your environment variables, setting it to cuda device 0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda:0"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20000" #Not useful to prevent memory crashes :(
print("Loading Vegvisir module from {}".format(vegvisir.__file__))


@Gooey(optional_cols=3,program_name="Vegvisir Executable with Pyinstaller",default_size=(1000,1000))
def parse_args(device,script_dir):
    #parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    parser = GooeyParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    args = VegvisirExample.parser_args(parser,device,script_dir)
    return args

#TODO: https://medium.com/codex/create-standalone-linux-installer-for-your-python-ai-application-5a31d99f9094

if __name__ == "__main__":
    args = parse_args(device,script_dir)
    if args.train:
        VegvisirExample.main(args)
    else:
        VegvisirAnalysis.analysis_models(args)
