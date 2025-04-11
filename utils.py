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

    factors = df[df["Unnamed: 0"] == case.name]

    return {
        "scale_drug_INa": factors["fINa "].values[0],
        "scale_drug_INaL": factors["fINaL "].values[0],
        "scale_drug_Ito": factors["fIto "].values[0],
        "scale_drug_ICaL": factors["fICaL"].values[0],
        "scale_drug_IKr": factors["fIKr "].values[0],
        "scale_drug_IKs": factors["fIKs"].values[0],
        "scale_drug_IK1": factors["fIK1 "].values[0],
    }
  

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
