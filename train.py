import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from preprocess import Preprocessor
import matplotlib.pyplot as plt
import numpy as np
from config import TitanicModel

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv("data/train.csv")
y = df["Survived"]
X = df.drop(columns=["Survived"])





preprocessor = Preprocessor()
preprocessor.fit(X)
X_processed = preprocessor.transform(X)

print(X_processed.head())


def normalize_column(df, col_name):
    """Normalize a DataFrame column to 0–1"""
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    if max_val == min_val:
        df[col_name] = 0.0
    else:
        df[col_name] = (df[col_name] - min_val) / (max_val - min_val)

numeric_cols = X_processed.select_dtypes(include=['float64', 'int64']).columns

print("Training...")
for col in numeric_cols:
    normalize_column(X_processed, col)


X_tensor = torch.tensor(X_processed.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)



X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)



model = TitanicModel(X_train.shape[1])

def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    best_loss = float("inf")
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "titanic_model.pt")

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        import pickle
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
    

    return losses

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds_class = (preds > 0.5).float()
        accuracy = (preds_class == y_test).float().mean().item()
    return accuracy

def plot_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

losses = train_model(model, X_train, y_train, X_test, y_test)
accuracy = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
plot_loss(losses)
