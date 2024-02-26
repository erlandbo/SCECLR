import time


def update_pbar(batch_idx, lendata):
    # https://stackoverflow.com/questions/3419984/print-to-the-same-line-and-not-a-new-line
    barlen = 30
    percentcomplete = (batch_idx + 1) / lendata * barlen
    percentremain = barlen - (batch_idx + 1) / lendata * barlen
    print("\r[{}]".format("*"*int(percentcomplete) + " " * int(percentremain)), end="")


def update_pbar_metrics(epoch, epoch_loss, lr, scores=None):
    print('Time:{} Epoch:{} LR:{} Loss:{} {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        epoch + 1,
        round(lr,5),
        round(epoch_loss, 5),
        " ".join([f"{model}:{scores}" for (model, scores) in scores.items()]) if scores is not None else ""
        )
    )


def get_expname(args):
    expname = f"{args.basedataset}"


if __name__ =="__main__":
    scores = {"KNN_score": 0.80, "SVM_score": 0.2}
    update_pbar_metrics(1, 0.4, scores)