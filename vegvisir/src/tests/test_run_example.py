import sys,os
import pytest
import logging
import subprocess
from pathlib import Path
logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parents[3] #two levels up dirname
script_dir = os.path.dirname(os.path.abspath(__file__))
#/home/dragon/drive/lys/Dropbox/PostDoc/vegvisir/Vegvisir_example.py


train_path = f"{EXAMPLE_DIR}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/variable_length_Icore_sequences_viral_dataset15_TRAIN.tsv"
test_path = f"{EXAMPLE_DIR}/vegvisir/src/vegvisir/data/benchmark_datasets/Icore/random_variable_length_Icore_sequences_viral_dataset15.tsv"

CPU_EXAMPLES = [f"{EXAMPLE_DIR}/Vegvisir_example.py -name viral_dataset15 -plot-all False",
                f"{EXAMPLE_DIR}Vegvisir_example.py -name custom_dataset -train-path {train_path} -test-path {test_path} -config-dict None -n 5 -train True -validate True -test False -plot-all False",
                ]
GPU_EXAMPLES = [f"{EXAMPLE_DIR}/Vegvisir_example.py -name viral_dataset15 -plot-all False",
                f"{EXAMPLE_DIR}/Vegvisir_example.py -name custom_dataset -train-path {train_path} -test-path {test_path} -config-dict None -n 5 -train True -validate True -test False -plot-all False",
               ]

@pytest.mark.skip(reason="skip")
@pytest.mark.parametrize("example", CPU_EXAMPLES)
def test_cpu_examples(example):
    logger.info("Running:\npython examples/{}".format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLE_DIR, filename)
    subprocess.check_call([sys.executable, filename] + args)

@pytest.mark.parametrize("example", GPU_EXAMPLES)
def test_gpu_examples(example):
    logger.info("Running:\npython examples/{}".format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLE_DIR, filename)
    subprocess.check_call([sys.executable, filename] + args)

if __name__ == '__main__':
    pytest.main(["-s",f"{script_dir}/test_run_example.py"]) #-s is to run the statements after yield?