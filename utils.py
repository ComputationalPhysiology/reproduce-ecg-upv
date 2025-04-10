from enum import IntEnum
from pathlib import Path


class Sex(IntEnum):
    undefined = 0
    male = 1
    female = 2


class Case(IntEnum):
    control = 0
    dofe = 1


def upv_path(sex: Sex, case: Case) -> Path:
    case2value = {
        Case.control: "ctrl",
        Case.dofe: "dofe",
    }

    fname = f"{sex.name}_{case2value[case]}.csv"

    return Path(__file__).absolute().parent / "results-upv" / fname


def case_parameters(case: Case) -> dict[str, float]:
    if case == Case.control:
        return {
            "scale_drug_INa": 1.0,
            "scale_drug_INaL": 1.0,
            "scale_drug_Ito": 1.0,
            "scale_drug_ICaL": 1.0,
            "scale_drug_IKr": 1.0,
            "scale_drug_IKs": 1.0,
            "scale_drug_IK1": 1.0,
        }
    elif case == Case.dofe:
        return {
            "scale_drug_INa": 1.0,
            "scale_drug_INaL": 1.0,
            "scale_drug_Ito": 0.75,
            "scale_drug_ICaL": 1.0,
            "scale_drug_IKr": 0.627,
            "scale_drug_IKs": 1.0,
            "scale_drug_IK1": 1.0,
        }
    else:
        raise ValueError(f"Unknown case {case}")


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
