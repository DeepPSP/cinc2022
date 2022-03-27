#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import os, sys

import numpy as np
import torch

from cfg import TrainCfg, ModelCfg

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print("Finding data files...")

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception("No data was provided.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ["Present", "Unknown", "Absent"]
    num_classes = len(classes)

    # TODO: Train the model.
    raise NotImplementedError

    if verbose >= 1:
        print("Done.")


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def load_challenge_model(model_folder, verbose):
    """ """
    raise NotImplementedError
    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def run_challenge_model(model, data, recordings, verbose):
    """ """
    raise NotImplementedError

    return classes, labels, probabilities
