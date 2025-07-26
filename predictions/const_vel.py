import pandas as pd
import numpy as np
from utils.baseline_config import FEATURE_FORMAT
import pickle as pkl

def get_mean_velocity(coords: np.ndarray) -> np.ndarray:
    vx, vy = np.zeros((coords.shape[0], coords.shape[1] - 1)), np.zeros((coords.shape[0], coords.shape[1] - 1))
    for t in range(1, coords.shape[1]):
        vx[:, t - 1] = (coords[:, t, 0] - coords[:, t - 1, 0]) / 0.1
        vy[:, t - 1] = (coords[:, t, 1] - coords[:, t - 1, 1]) / 0.1

    vx_mean = np.mean(vx, axis=1)
    vy_mean = np.mean(vy, axis=1)
    return vx_mean, vy_mean


def predict(obs_trajectory: np.ndarray, vx_mean: np.ndarray, vy_mean: np.ndarray, pred_len: int) -> np.ndarray:
    pred_traj = np.zeros((obs_trajectory.shape[0], pred_len, 2))
    prev_coords = obs_trajectory[:, -1, :]
    for t in range(pred_traj.shape[1]):
        pred_traj[:, t, 0] = prev_coords[:, 0] + vx_mean * 0.1
        pred_traj[:, t, 1] = prev_coords[:, 1] + vy_mean * 0.1
        prev_coords = pred_traj[:, t, :]

    return pred_traj


if __name__ == "__main__":
    FEATURES_PATH = "../features/forecasting_features_train.pkl"
    PREDICTIONS_PATH = "../pred_trajectory/pred_trajectory_train.pkl"

    obs_len = 20
    pred_len = 30

    df = pd.read_pickle(FEATURES_PATH)
    seq_id = df["SEQUENCE"].values
    obs_trajectory = np.stack(df["FEATURES"].values)[:, :obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
        "float")

    vx, vy = get_mean_velocity(obs_trajectory)
    pred_trajectory = predict(obs_trajectory, vx, vy, pred_len)

    forecasted_traj = {}
    for i in range(seq_id.shape[0]):
        forecasted_traj[seq_id[i]] = [pred_trajectory[i, :, :]]

    with open(PREDICTIONS_PATH, "wb") as f:
        pkl.dump(forecasted_traj, f)