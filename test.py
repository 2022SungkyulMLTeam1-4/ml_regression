import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LifeExpectationDataset(Dataset):
    """
    기대수명 예측을 위한 데이터셋 클래스입니다.
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


dataset = LifeExpectationDataset()

model = torch.load("best_model.pt")

with torch.no_grad():
    model.eval()
    y_pred = model(dataset[:][0].to(device))
    val_loss = torch.nn.functional.mse_loss(y_pred, dataset[:][1].to(device))

print(val_loss)
