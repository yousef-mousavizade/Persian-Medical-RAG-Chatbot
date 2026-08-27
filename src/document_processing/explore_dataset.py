
import pandas as pd

DATA_PATH = r"D:\Programing\Github\Persian-Medical-RAG-Chatbot\data\raw\Specialized_Dataset.xlsx"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Total number of records : {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Name of columns: {list(df.columns)}")
    return df


def basic_report(df: pd.DataFrame) -> None:
    print("\n=== Number of empty values per column ===")
    print(df.isnull().sum())

    print("\n=== Records with answers ===")
    has_response = df["response"].notna().sum()
    print(f"{has_response} out of {len(df)} records have answer")

    print("\n=== Number of unique drugs ===")
    print(df["drug_name"].nunique())

    print("\n=== Example of a record ===")
    sample = df[df["response"].notna()].iloc[0]
    print("Drug :", sample["drug_name"])
    print("Question:", sample["comment"])
    print("Response:", sample["response"])


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    basic_report(df)

