import os
import glob
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='tb_logs_writer')
parser.add_argument('--logdir', default="logs", type=str)


def write_summary(logpath="logs"):
    for filepath in glob.glob(f'{logpath}/*/train.log'):
        print(filepath)
        name = filepath.split("/")[-2]
        print(name)
        tb_log_path = "tb_logs/" + name
        #if os.path.exists(tb_log_path): continue

        writer = SummaryWriter("tb_logs/" + name)
        with open(filepath, 'r') as f:
            for line in f.readlines():
                print(line)
                metrics = {l.split(":")[0] : float(l.split(":")[1]) for i, l in enumerate( line.split(" ") ) if l.strip() and i > 1}
                #print(metrics.keys())
                if metrics:
                    stage, epoch = int(metrics["Stage"]), int(metrics["Epoch"])
                    for k, v in metrics.items():
                        writer.add_scalar(f"stage{stage}_{k}", v, global_step=epoch)
                    #print(metrics)


if __name__ == '__main__':
    args = parser.parse_args()

    write_summary(args.logdir)
