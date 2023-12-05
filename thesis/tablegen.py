import pandas as pd
from beartype import beartype as typed

csv = pd.read_csv("thesis/finetuning.csv", header=None)


@typed
def add_formatting(series: pd.Series) -> list[str]:
    arr = list(map(float, series))
    assert len(arr) == 4
    outputs = []
    for i in range(4):
        if i == 3:
            outputs.append(f"{arr[i]:.2f}")
        elif arr[i] < arr[-1] + 0.2:
            outputs.append(f"*{arr[i]:.2f}*")
        else:
            outputs.append(f"{arr[i]:.2f}")
    return outputs


for idx, row in csv.iterrows():
    outputs = [row[0], row[1]]
    outputs.extend(add_formatting(row[2:6]))
    outputs.extend(add_formatting(row[6:]))
    print(",".join(f"[{elem}]" for elem in outputs), end=",\n")
