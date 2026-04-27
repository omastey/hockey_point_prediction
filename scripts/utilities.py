import pandas as pd

def preview_dataset(path, n_head=5, n_tail=5):
    """
    Loads a Parquet dataset and prints shape, columns,
    head, and tail for quick inspection.
    """
    df = pd.read_parquet(path)

    print("=" * 60)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 60)

    print("\n🧾 Columns:")
    print(df.columns.tolist())

    print("\n🔼 Head:")
    print(df.head(n_head))

    print("\n🔽 Tail:")
    print(df.tail(n_tail))

    return df

if __name__ == "__main__":
    dataset_path = "edge_data/nhl_merged_dataset.parquet"
    preview_dataset(dataset_path)