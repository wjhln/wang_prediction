import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="MLP prediction")

    parser.add_argument('--train_features', default='features/forecasting_features_train.pkl', type=str,
                        help='path to train features')
    parser.add_argument('--val_features', default='features/forecasting_features_val.pkl', type=str,
                        help='path to validation features')
    parser.add_argument('--test_features', default='features/forecasting_features_test.pkl', type=str,
                        help='path to test features')

    parser.add_argument('--model_path', defult='', type=str, helper='path to model save')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')

    return parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    args = parse_arguments()


if __name__ == "__main__":
    main()
