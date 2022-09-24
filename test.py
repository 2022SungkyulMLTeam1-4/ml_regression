import pandas as pd
import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LifeExpectationDataset(Dataset):
    """
    기대수명 예측을 위한 데이터셋 클래스입니다.
    """

    def __init__(self):
        # 연도, 국가를 제외한 이유는 연도와 국가의 수치/범주 요소 자체를 학습할 위험이 있기 때문입니다.
        # 개발도상국 여부도 직접적인 예측에 도움이 되지 않는 것이, 일례로 한국의 경우 개발도상국으로 분류되어 있지만
        # 다양한 요소를 고려할 경우 선진국과 크게 다른 요소가 있다고 판단하기 힘듭니다.
        # 따라서 이러한 outlier가 등장할 확률이 높으므로, 개발도상국 여부는 예측을 더 어렵게 할 위험이 존재합니다.
        # 인구와 GDP는 결측치가 많음과 동시에 다른 요소로부터 간접적으로 산출 가능하며
        # 이 자체로 예측에 도움이 되지 않을 것이라 판단하였습니다.
        data = pd.read_csv("Life Expectancy Data.csv").drop(
            ["Country", "Status", "Year", "Population", "GDP"], axis=1
        )
        data.dropna(axis=0, inplace=True)
        self.x_data = data.drop(["Life expectancy"], axis=1).to_numpy()
        self.y_data = data[["Life expectancy"]].to_numpy()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data[index])
        y = torch.FloatTensor(self.y_data[index])
        return x, y


dataset = LifeExpectationDataset()
model: torch.nn.Linear = torch.load("best_model.pt")

with torch.no_grad():
    model.eval()
    y_pred = model(dataset[:][0].to(device))
    val_loss = torch.nn.functional.mse_loss(y_pred, dataset[:][1].to(device))

print(val_loss)
