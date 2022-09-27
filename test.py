import pandas as pd
import sklearn.metrics
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LifeExpectationDataset(Dataset):
    """
    기대수명 예측을 위한 데이터셋 클래스입니다.

    https://wikidocs.net/57165 사이트를 참조하여 데이터셋 구축을 수행하였습니다.
    """

    def __init__(self):
        data = pd.read_csv("Life Expectancy Data.csv").drop(
            ["Country", "Status", "Year", "Population", "GDP"], axis=1
        )
        data.dropna(axis=0, inplace=True)
        scaler = StandardScaler()
        self.x_data = scaler.fit_transform(
            data.drop(["Life expectancy"], axis=1).to_numpy()
        )
        self.y_data = data[["Life expectancy"]].to_numpy()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data[index])
        y = torch.FloatTensor(self.y_data[index])
        return x, y


def random_split_train_test(data, train_ratio=0.8):
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    return torch.utils.data.random_split(data, [train_size, test_size])


if __name__ == "__main__":
    dataset = LifeExpectationDataset()
    _, test_dataset = random_split_train_test(dataset, 0.8)

    model = torch.load("best_model.pt")

    with torch.no_grad():
        model.eval()
        y_pred = model(test_dataset[:][0].to(device))
        y_true = test_dataset[:][1].to(device)
        val_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        r2_score = sklearn.metrics.r2_score(y_true.cpu(), y_pred.cpu())
    print(val_loss)
    print("결정계수:", r2_score)
