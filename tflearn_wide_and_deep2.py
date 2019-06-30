
from __future__ import division, print_function

import os
import sys
import argparse
# import zqtflearn
import tempfile
import urllib

import numpy as np
# import pandas as pd
import tensorflow as tf

#-----------------------------------------------------------------------------

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = {"workclass": 10, "education": 17, "marital_status":8, 
                       "occupation": 16, "relationship": 7, "race": 6, 
                       "gender": 3, "native_country": 43, "age_binned": 14}

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def CommandLine(args=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    if args:
        print("added by zhengquan. I think it didn't get here, test it tomorrow")
        FLAGS.__init__()
        FLAGS.__dict__.update(args)

    try:
        flags.DEFINE_string("model_type", "wide+deep","Valid model types: {'wide', 'deep', 'wide+deep'}.")
        flags.DEFINE_string("run_name", None, "name for this run (defaults to model type)")
        flags.DEFINE_string("load_weights", None, "filename with initial weights to load")
        flags.DEFINE_string("checkpoints_dir", None, "name of directory where checkpoints should be saved")
        flags.DEFINE_integer("n_epoch", 200, "Number of training epoch steps")
        flags.DEFINE_integer("snapshot_step", 100, "Step number when snapshot (and validation testing) is done")
        flags.DEFINE_float("wide_learning_rate", 0.001, "learning rate for the wide part of the model")
        flags.DEFINE_float("deep_learning_rate", 0.001, "learning rate for the deep part of the model")
        flags.DEFINE_boolean("verbose", False, "Verbose output")
    except argparse.ArgumentError:
        pass	# so that CommandLine can be run more than once, for testing
    print("FLAGS.verbose")
    print(FLAGS.verbose)


#-----------------------------------------------------------------------------

if __name__=="__main__":
    CommandLine()
    None
