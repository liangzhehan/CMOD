from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error


def eval_od_prediction(model, data, od_matrix, back_points, st, ed, device, config):
    input_len = config["input_len"]
    output_len = config["output_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    label, prediction = [], []

    with torch.no_grad():
        model = model.eval()
        num_test_batch = (ed - st - output_len) // input_len

        for j in tqdm(range(num_test_batch)):
            begin_time = j * input_len + st
            now_time = j * input_len + input_len + st
            if now_time % day_cycle < day_start or now_time % day_cycle > day_end:
                continue
            head, tail = back_points[begin_time // input_len], back_points[
                now_time // input_len]  # [head,tail1) nowtime [tail1,tail2) nowtime+τ

            if head == tail:
                continue

            if now_time % day_cycle >= day_end:
                predict_od = False
            else:
                predict_od = True

            sources_batch, destinations_batch = data.sources[head:tail], data.destinations[head:tail]
            timestamps_batch_torch = torch.Tensor(data.timestamps[head:tail]).to(device)
            time_diffs_batch_torch = torch.Tensor(data.timestamps[head:tail] - now_time).to(device)

            od_matrix_predicted = model.compute_od_matrix(
                sources_batch, destinations_batch, timestamps_batch_torch,
                time_diffs_batch_torch, now_time, begin_time,
                predict_od=predict_od)

            if predict_od:
                prediction.append(od_matrix_predicted.cpu().numpy())
                od_matrix_real = od_matrix[now_time // input_len]
                label.append(od_matrix_real)
        stacked_prediction = np.stack(prediction)
        stacked_prediction[stacked_prediction < 0] = 0
        stacked_label = np.stack(label)
        reshaped_prediction = stacked_prediction.reshape(-1)
        reshaped_label = stacked_label.reshape(-1)
        mse = mean_squared_error(reshaped_prediction, reshaped_label)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(reshaped_prediction, reshaped_label)
        pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
        smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
                np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
        print(mse, mae, pcc, smape)

    return mse, rmse, mae, pcc, smape, stacked_prediction, stacked_label
