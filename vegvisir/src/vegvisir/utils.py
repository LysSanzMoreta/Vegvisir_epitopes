import argparse
import ast
def str2bool(v):
    """Converts a string into a boolean, useful for boolean arguments
    :param str v"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2None(v):
    """Converts a string into None
    :param str v"""

    if v.lower() in ('None'):
        return None
    else:
        v = ast.literal_eval(v)
        return v