import sys
import time
import logging
import os
import json


def initialize_logger(args):
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    exppath = os.path.join(args.mainpath, "logs/"+current_time + f"_{args.basedataset}_{args.criterion}_{args.metric}_batch{args.batchsize}_feature{args.mlp_outfeatures}/")
    os.makedirs(exppath, exist_ok=False)
    args.exppath = exppath
    logging.basicConfig(format='%(message)s', filename=exppath + "/train.log", level=logging.INFO)

    return logging


def store_hyperparameters(args, tag=""):
    with open(args.exppath + f"/hyperparameters{tag}.txt", "w") as f:
        f.write(json.dumps(vars(args)))


def read_hyperparameters(filepath):
    with open(filepath, "r") as f:
        hyperparams = json.load(f)
    return hyperparams
