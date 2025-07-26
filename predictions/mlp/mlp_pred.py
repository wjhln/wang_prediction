import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Using device: ", device)

if __name__ == "__main__":
    FEATURES_PATH = "../features/forecasting_features_train.pkl"
    PREDICTIONS_PATH = "../pred_trajectory/pred_trajectory_train.pkl"
