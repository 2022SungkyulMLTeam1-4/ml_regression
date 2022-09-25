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


if __name__ == "__main__":
    dataset = LifeExpectationDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_size, shuffle=True
    )

    criterion = torch.nn.functional.mse_loss

    model = torch.nn.Linear(16, 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    least_val_loss = 1e9
    total_epochs = 100000
    for epoch in range(1, total_epochs):
        final_cost = None
        for batch_index, samples in enumerate(train_dataloader):
            x_train, y_train = samples
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            prediction = model(x_train)
            cost = criterion(prediction, y_train)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            final_cost = cost

        with torch.no_grad():
            model.eval()
            y_pred = model(test_dataset[:][0].to(device))
            new_val_loss = criterion(y_pred, test_dataset[:][1].to(device))

            if new_val_loss < least_val_loss:
                torch.save(model, "best_model.pt")
                least_val_loss = new_val_loss

        print(
            "Epoch %4d/%4d Loss: %20.4f val_loss: %20.4f"
            % (epoch + 1, total_epochs, final_cost.item(), new_val_loss)
        )
