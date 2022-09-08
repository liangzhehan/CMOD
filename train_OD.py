import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random
from tqdm import trange
import shutil
from evaluation.evaluation import eval_od_prediction
from model.CMOD import CMOD
from utils.utils import EarlyStopMonitor
from utils.data_processing import get_od_data

config = {
    "NYTaxi": {
        "data_path": "data/NYTaxi/NYTaxi.npy",
        "matrix_path": "data/NYTaxi/od_matrix.npy",
        "point_path": "data/NYTaxi/back_points.npy",
        "input_len": 1800,
        "output_len": 1800,
        "day_cycle": 86400,
        "train_day": 139,
        "val_day": 21,
        "test_day": 21,
        "day_start": 0,
        "day_end": 86400,
        "n_nodes": 63
    }
}

### Argument and global variables
parser = argparse.ArgumentParser('CMOD training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. NYTaxi or BJSubway)',
                    default='NYTaxi')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cpu", help='Idx for the gpu to use: cpu, cuda:0, etc.')

parser.add_argument('--loss', type=str, default="odloss", help='Loss function')
parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for '
                                                               'each node')
parser.add_argument('--lambs', type=float, nargs="+", default=[1], help='Lamb of different time scales')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

NUM_EPOCH = args.n_epoch
device = args.device
DATA = args.data
LEARNING_RATE = args.lr
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

input_len = config[DATA]["input_len"]
output_len = config[DATA]["output_len"]
day_cycle = config[DATA]["day_cycle"]
day_start = config[DATA]["day_start"]
day_end = config[DATA]["day_end"]

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.data}-{args.suffix}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.data}-{args.suffix}-{epoch}.pth'
results_path = f"results/{args.data}_{args.suffix}.pkl"
Path("results/").mkdir(parents=True, exist_ok=True)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f"log/{str(time.time())}_{args.suffix}.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points = get_od_data(config[DATA])
model = CMOD(device=device, n_nodes=n_nodes, node_features=node_features,
             message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
             output=output_len, lambs=args.lambs)


class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.pro = torch.nn.ReLU()

    def forward(self, predict, truth):
        mask = (truth < 1)
        mask2 = (predict > 0)
        loss = torch.mean(((predict - truth) ** 2) * ~mask + ((mask2 * predict - truth) ** 2) * mask)
        return loss


if args.loss == "odloss":
    logger.info("od loss!!!!!")
    criterion = OD_loss()
else:
    criterion = torch.nn.MSELoss()
    logger.info("mse loss!!!!!")

model = model.to(device)

val_mses = []
epoch_times = []
total_epoch_times = []
train_losses = []
if args.best == "":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
    num_batch = (val_time - input_len) // input_len
    for epoch in range(NUM_EPOCH):
        print("================================Epoch: %d================================" % epoch)
        start_epoch = time.time()
        logger.info('start {} epoch'.format(epoch))
        m_loss = []

        model.init_memory()
        model = model.train()
        batch_range = trange(num_batch)
        for j in batch_range:
            ### Training
            begin_time = j * input_len
            now_time = j * input_len + input_len
            if now_time % day_cycle < day_start or now_time % day_cycle > day_end:
                continue

            head, tail = back_points[begin_time // input_len], back_points[
                now_time // input_len]  # [head,tail1) nowtime [tail1,tail2) nowtime+τ
            if head == tail:
                continue

            time_of_matrix = now_time % day_cycle // input_len
            weekday_of_matrix = now_time // day_cycle % 7
            time_of_matrix2 = (now_time + input_len) % day_cycle // input_len
            weekday_of_matrix2 = (now_time + input_len) // day_cycle % 7

            optimizer.zero_grad()
            sources_batch, destinations_batch = full_data.sources[head:tail], full_data.destinations[head:tail]
            edge_idxs_batch = full_data.edge_idxs[head:tail]
            timestamps_batch_torch = torch.Tensor(full_data.timestamps[head:tail]).to(device)
            time_diffs_batch_torch = torch.Tensor(full_data.timestamps[head:tail] - now_time).to(device)

            if now_time % day_cycle >= day_end:
                predict_od = False
            else:
                predict_od = True

            # Predict OD, get updated memories and messages
            od_matrix_predicted = model.compute_od_matrix(
                sources_batch, destinations_batch, timestamps_batch_torch,
                time_diffs_batch_torch, now_time, begin_time,
                predict_od=predict_od)

            if predict_od:
                od_matrix_real = od_matrix[now_time // input_len]
                loss = criterion(od_matrix_predicted, torch.FloatTensor(od_matrix_real).to(device))
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

            batch_range.set_description(f"train_loss: {m_loss[-1]};")

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        print("================================Val================================")
        val_mse, val_rmse, val_mae, val_pcc, val_smape, _, _ = eval_od_prediction(model=model,
                                                                                  data=full_data,
                                                                                  od_matrix=od_matrix,
                                                                                  back_points=back_points,
                                                                                  st=val_time,
                                                                                  ed=test_time,
                                                                                  device=device,
                                                                                  config=config[DATA])

        val_mses.append(val_mse)
        train_losses.append(np.mean(m_loss))
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        # Save temporary results
        pickle.dump({
            "val_mses": val_mses,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
        logger.info(
            f'Epoch val metric: mae, mse, rmse, pcc, smape, {val_mae}, {val_mse}, {np.sqrt(val_mse)}, {val_pcc}, {val_smape}')
        # Early stopping
        ifstop, ifimprove = early_stopper.early_stop_check(val_mse)
        if ifstop:
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(
                {"statedict": model.state_dict(), "memory": model.backup_memory()},
                get_checkpoint_path(epoch))
    logger.info('Saving DyOD model')
    shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
    logger.info('DyOD model saved')
    best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
else:
    best_model_param = torch.load(args.best)

# load model parameters, memories from best epoch on val dataset
model.load_state_dict(best_model_param["statedict"])
model.restore_memory(best_model_param["memory"])

# Test
print("================================Test================================")
test_mse, test_rmse, test_mae, test_pcc, test_smape, prediction, label = eval_od_prediction(model=model,
                                                                                            data=full_data,
                                                                                            od_matrix=od_matrix,
                                                                                            back_points=back_points,
                                                                                            st=test_time,
                                                                                            ed=all_time,
                                                                                            device=device,
                                                                                            config=config[DATA])

logger.info(
    'Test statistics:-- mae: {}, mse: {}, rmse: {}, pcc: {}, smape:{}'.format(test_mae, test_mse, test_rmse, test_pcc,
                                                                              test_smape))
# Save results for this run
pickle.dump({
    "val_mses": val_mses,
    "test_mse": test_mse,
    "test_rmse": np.sqrt(test_mse),
    "test_mae": test_mae,
    "test_pcc": test_pcc,
    "test_smape": test_smape,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times,
    "prediction": prediction,
    "label": label
}, open(results_path, "wb"))
