import sys
import time
import logging
import os


def update_pbar(batch_idx, lendata):
    # https://stackoverflow.com/questions/3419984/print-to-the-same-line-and-not-a-new-line
    barlen = 30
    percentcomplete = (batch_idx + 1) / lendata * barlen
    percentremain = barlen - (batch_idx + 1) / lendata * barlen
    print("\r[{}]".format("*"*int(percentcomplete) + " " * int(percentremain)), end="")


# def update_pbar_metrics(epoch, epoch_loss, lr, scores=None):
#     print('Time:{} Epoch:{} LR:{} Loss:{} {}'.format(
#         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#         epoch + 1,
#         round(lr,5),
#         round(epoch_loss, 5),
#         " ".join([f"{model}:{scores}" for (model, scores) in scores.items()]) if scores is not None else ""
#         )
#     )


def update_log(trainlogger, stage, epoch, epoch_loss, lr, scores=None):
    log_str = 'Time:{} Stage:{} Epoch:{} LR:{} Loss:{} {}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            stage,
            epoch,
            lr,
            epoch_loss,
            " ".join([f"{model}:{scores}" for (model, scores) in scores.items()]) if scores is not None else ""
        )
    # PBar terminal
    print(log_str)
    # Logger
    trainlogger.info(log_str)


def write_model(model, args):
    output = args.exppath + "/models.txt"
    with open(output, "a") as f:
        sys.stdout = f
        print(model)
    sys.stdout = sys.__stdout__


def initialize_logger(args):
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    exppath = os.path.join(args.mainpath, "logs/"+current_time)
    os.makedirs(exppath, exist_ok=False)
    args.exppath = exppath
    logging.basicConfig(format='%(message)s', filename=exppath + "/train.log", level=logging.INFO)

    write_parameters(args)
    return logging


# TODO serialize
def write_parameters(args):
    with open(args.exppath + "/parameters.txt", "w") as f:
        for key, value in args.__dict__.items():
            f.write(f"{key}: {value}\n")


if __name__ =="__main__":
    scores = {"KNN_score": 0.80, "SVM_score": 0.2}
