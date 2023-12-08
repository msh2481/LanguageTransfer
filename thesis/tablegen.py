import pandas as pd
from beartype import beartype as typed

@typed
def finetuning_table() -> None:
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

    df = pd.read_csv("thesis/finetuning.csv", header=None)
    for idx, row in df.iterrows():
        outputs = [row[0], row[1]]
        outputs.extend(add_formatting(row[2:6]))
        outputs.extend(add_formatting(row[6:]))
        print(",".join(f"[{elem}]" for elem in outputs), end=",\n")

@typed
def probes_table(filename: str) -> None:
    df = pd.read_json(filename)
    df.loc["average"] = df.mean()
    model_names = [""] + list(df.columns)
    for name in model_names:
        if name:
            print(f"[*{name}*],", end="")
        else:
            print("[],", end="")
    print()
    for idx, row in df.iterrows():
        if idx == "average":
            print("[*Average*],", end="")
        else:
            print(f"[#text(`{idx}`, 7pt)],", end="")
        for value in row:
            print(f"[{value:.2f}],", end="")
        print()


probes_table("thesis/cloze.json")
