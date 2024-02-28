import os
import glob
from torch.utils.tensorboard import SummaryWriter


def write_summary():
    for filepath in glob.glob('logs/*/train.log'):
        print(filepath)
        name = filepath.split("/")[-2]
        print(name)
        tb_log_path = "tb_logs/" + name
        if os.path.exists(tb_log_path): continue

        writer = SummaryWriter("tb_logs/" + name)
        with open(filepath, 'r') as f:
            for line in f.readlines():
                metrics =  {l.split(":")[0] : float(l.split(":")[1]) for i, l in enumerate( line.split(" ") ) if l and i > 1}
                print(metrics.keys())
                if metrics:
                    stage, epoch = int(metrics["Stage"]), int(metrics["Epoch"])
                    for k, v in metrics.items():
                        writer.add_scalar(f"stage{stage}_{k}", v, global_step=epoch)
                    #print(metrics)


if __name__ == '__main__':
    write_summary()