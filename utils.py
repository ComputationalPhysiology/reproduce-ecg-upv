from enum import IntEnum
from pathlib import Path


class Sex(IntEnum):
    undefined = 0
    male = 1
    female = 2


class Case(IntEnum):
    CTRL = 0
    Astemizole = 1
    Azimilide = 2
    Bepridil = 3
    Chlorpromazine = 4
    Cisapride = 5
    Clarithromycin = 6
    Clozapine = 7
    Diltiazem = 8
    Disopyramide = 9
    Dofetilide = 10
    Domperidone = 11
    Droperidol = 12
    Ibutilide = 13
    Loratadine = 14
    Metoprolol = 15
    Mexiletine = 16
    Nifedipine = 17
    Nitrendipine = 18
    Ondansetron = 19
    Pimozide = 20
    Quinidine = 21
    Ranolazine = 22
    Risperidone = 23
    Sotalol = 24
    Tamoxifen = 25
    Terfenadine = 26
    Vandetanib = 27
    Verapamil = 28



def upv_path(sex: Sex, case: Case) -> Path:
    case2value = {
        Case.CTRL: "ctrl",
        Case.Dofetilide: "dofe",
    }

    fname = f"{sex.name}_{case2value[case]}.csv"

    return Path(__file__).absolute().parent / "results-upv" / fname


def case_parameters(case: Case) -> dict[str, float]:
    import pandas as pd

    df = pd.read_excel(
        Path(__file__).absolute().parent / "CiPA28.xlsx",
        sheet_name="fraction",
    )

    nan_or_one = lambda x: 1.0 if pd.isna(x) else float(x)

    factors = df[df["Unnamed: 0"] == case.name]

    values= {
        "scale_drug_INa": nan_or_one(factors["fINa "].values[0]),
        "scale_drug_INaL": nan_or_one(factors["fINaL "].values[0]),
        "scale_drug_Ito": nan_or_one(factors["fIto "].values[0]),
        "scale_drug_ICaL": nan_or_one(factors["fICaL"].values[0]),
        "scale_drug_IKr": nan_or_one(factors["fIKr "].values[0]),
        "scale_drug_IKs": nan_or_one(factors["fIKs"].values[0]),
        "scale_drug_IK1": nan_or_one(factors["fIK1 "].values[0]),
    }
    return values
  

def get_lead_positions() -> dict[str, tuple[float, float, float]]:
    def name2lead(name):
        if "LA" in name:
            return "LA"
        if "RA" in name:
            return "RA"
        if "LL" in name:
            return "LL"
        raise ValueError(f"Unknown lead name: {name}")

    text = Path("limb_position.txt").read_text()
    leads = {}
    for line in text.splitlines():
        pos, name = line.strip().split("!")
        leads[name2lead(name)] = tuple(float(p) for p in pos.strip().split(" "))

    return leads
