import sys,os
import pytest
import logging
import subprocess
from pathlib import Path
logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parents[2] #two levels up dirname

#TODO: Add tests:
# a) Custom example load,
# b) build True example
# c) Infer angles
CPU_EXAMPLES = ["Vegvisir_example.py -name viral_dataset15",
                "Vegvisir_example.py -name custom_dataset -train_path vegvisir/src/vegvisir/data/benchmark_dataset/Icore/variable_length_Icore_sequences_viral_dataset15_TRAIN.tsv vegvisir/src/vegvisir/data/benchmark_dataset/Icore/random_variable_length_Icore_sequences_viral_dataset15.tsv -conig-dict None -n 5 -train True -validate -test False",
                "Vegvisir_example.py ",
                "Vegvisir_example.py ",
                ]
GPU_EXAMPLES = ["Vegvisir_example.py -name viral_dataset15",
                "Vegvisir_example.py -name custom_dataset",
                "Vegvisir_example.py -name custom_dataset",
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
    pytest.main(["-s","test_examples.py"]) #-s is to run the statements after yield?