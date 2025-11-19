import pandas as pd
from pprint import pprint

# For reference
if __name__ == "__main__":
    file = "largerExample"
    data = pd.read_csv(f"data/{file}.csv")
    pprint(data)
    data["drainLength"] = data["drainLength"] / 2
    model(data)
    pprint(data)
