import pandas as pd
from config import SEX_MAP, EMBARKED_MAP


class Preprocessor:
    def __init__(self):
        self.age_median = None
        self.fare_median = None
        self.embarked_mode = None
        self.min_max = {}  

    # -------- TRAINING TIME --------
    def fit(self, df: pd.DataFrame):
        """Learn statistics from training data"""
        self.age_median = df["Age"].median()
        self.fare_median = df["Fare"].median()
        self.embarked_mode = df["Embarked"].mode()[0]

        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            self.min_max[col] = (df[col].min(), df[col].max())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to a dataframe"""
        df = df.copy()

        # Missing values
        df["Age"] = df["Age"].fillna(self.age_median)
        df["Fare"] = df["Fare"].fillna(self.fare_median)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)

        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df.drop(columns=["Cabin", "Ticket", "Name"], inplace=True, errors="ignore")

        # Encoding
        df["Sex"] = df["Sex"].map(SEX_MAP)
        df["Embarked"] = df["Embarked"].map(EMBARKED_MAP)

        # Final cleanup
        df = df.drop(columns=["PassengerId"], errors="ignore")

        return df

    # -------- INFERENCE TIME --------
    def transform_single(self, data: dict) -> pd.DataFrame:
        """Preprocess a single passenger input"""
        df = pd.DataFrame([{
            "Pclass": data["Pclass"],
            "Sex": SEX_MAP.get(data["Sex"], -1),
            "Age": data.get("Age", self.age_median),
            "SibSp": data["SibSp"],
            "Parch": data["Parch"],
            "Fare": data.get("Fare", self.fare_median),
            "Embarked": EMBARKED_MAP.get(data.get("Embarked"), self.embarked_mode),
            "HasCabin": int(data.get("Cabin") is not None)
        }])

        return df
