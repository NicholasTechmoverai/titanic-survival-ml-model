import torch
from preprocess import Preprocessor
from config import TitanicModel
import pickle
import pandas as pd

device = "cpu"

model = TitanicModel(input_dim=8)
model.load_state_dict(torch.load("titanic_model.pt", map_location=device))
model.eval()

with open("preprocessor.pkl", "rb") as f:
    preprocessor: Preprocessor = pickle.load(f)


sample_user_data = [
    {
        "Name": "Braund, Mr. Owen Harris",
        "Ticket": "A/5 21171",
        "Pclass": 1,
        "Sex": "female",
        "Age": 38,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.2833,
        "Embarked": "C",
        "Cabin": "C85"
    },
    {
        "Name": "Heikkinen, Miss. Laina",
        "Ticket": "STON/O2. 3101282",
        "Pclass": 3,
        "Sex": "male",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.925,
        "Embarked": "S",
        "Cabin": None
    },
    {
        "Name": "Oliva y Ocana, Dona. Fermina",
        "Ticket": "PC 17758",
        "Pclass": 2,
        "Sex": "female",
        "Age": 45,
        "SibSp": 1,
        "Parch": 1,
        "Fare": 30.0708,
        "Embarked": "Q",
        "Cabin": "D15"
    },
    {
        "Name": "Ware, Mr. Frederick",
        "Ticket": "359309",
        "Pclass": 3,
        "Sex": "male",
        "Age": 60,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.05,
        "Embarked": "S",
        "Cabin": None
    },
    {
        "Name": "Johnston, Mr. Alfred",
        "Ticket": "113803",
        "Pclass": 1,
        "Sex": "male",
        "Age": 50,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 53.1,
        "Embarked": "C",
        "Cabin": "B28"
    }
]

def normalize_user_input(df, training_stats):
    """Normalize numerical columns using training data min-max values"""
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        min_val, max_val = training_stats[col]
        df[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.0
    return df

# Compute min-max from training data (from preprocessor if needed)
# i've saved them in preprocessor
training_stats = {
    "Pclass": (preprocessor.min_max.get("Pclass", (1,3))),
    "Age": (preprocessor.min_max.get("Age", (0, 80))),
    "SibSp": (preprocessor.min_max.get("SibSp", (0,8))),
    "Parch": (preprocessor.min_max.get("Parch", (0,6))),
    "Fare": (preprocessor.min_max.get("Fare", (0,512)))
}

def test(user_data):
    X_user = preprocessor.transform_single(user_data)
    X_user = normalize_user_input(X_user, training_stats)
    X_tensor = torch.tensor(X_user.values, dtype=torch.float32, device=device)


    with torch.no_grad():
        survival_prob = model(X_tensor).item()

    return f"{'-'*20}\nName: {user_data['Name']} | Ticket: {user_data['Ticket']} | Survival prob: {survival_prob:.2%}"

for i in sample_user_data:
    print(test(i))
