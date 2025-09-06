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
    Quinidine_TdP = 29
    Clozapine_TdP = 30


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


def profile_parameters(profile_number: int) -> dict[str, float]:

    if profile_number == 1:
        return {
            "scale_sex_GNa_male": 1.4378,
            "scale_sex_GNaL_male": 1.4378,
            "scale_sex_Gtos_male": 1.3478,
            "scale_sex_GCaL_male": 0.8462,
            "scale_sex_GKr_male": 0.8638,
            "scale_sex_GKs_male": 1.0578,
            "scale_sex_GK1_male": 1.0628,
            "scale_sex_GNCX_male": 1.0253,
            "scale_sex_PNaK_male": 0.3018,
            "scale_sex_GKb_male": 0.7998,
            "scale_sex_GpCa_male": 0.9706,
            "scale_sex_GJup_male": 0.9594,
            "scale_sex_calm_male": 1.0496,
            "scale_sex_GNa_female": 1.1395,
            "scale_sex_GNaL_female": 1.1395,
            "scale_sex_Gtos_female": 0.8499,
            "scale_sex_GCaL_female": 1.5798,
            "scale_sex_GKr_female": 0.9298,
            "scale_sex_GKs_female": 0.791,
            "scale_sex_GK1_female": 0.5367,
            "scale_sex_GNCX_female": 0.8128,
            "scale_sex_PNaK_female": 0.2958,
            "scale_sex_GKb_female": 1.7695,
            "scale_sex_GpCa_female": 1.0027,
            "scale_sex_GJup_female": 0.9383,
            "scale_sex_calm_female": 1.4384,
        }
    elif profile_number == 2:
        return {
            "scale_sex_GNa_male": 1.0705,
            "scale_sex_GNaL_male": 1.0705,
            "scale_sex_Gtos_male": 1.3165,
            "scale_sex_GCaL_male": 1.4489,
            "scale_sex_GKr_male": 0.9385,
            "scale_sex_GKs_male": 0.975,
            "scale_sex_GK1_male": 0.6759,
            "scale_sex_GNCX_male": 0.7469,
            "scale_sex_PNaK_male": 0.3983,
            "scale_sex_GKb_male": 1.2688,
            "scale_sex_GpCa_male": 1.6286,
            "scale_sex_GJup_male": 1.5045,
            "scale_sex_calm_male": 1.1102,
            "scale_sex_GNa_female": 0.6591,
            "scale_sex_GNaL_female": 0.6591,
            "scale_sex_Gtos_female": 0.9357,
            "scale_sex_GCaL_female": 1.3217,
            "scale_sex_GKr_female": 0.658,
            "scale_sex_GKs_female": 0.8026,
            "scale_sex_GK1_female": 0.8265,
            "scale_sex_GNCX_female": 0.5792,
            "scale_sex_PNaK_female": 0.4249,
            "scale_sex_GKb_female": 1.3496,
            "scale_sex_GpCa_female": 1.3057,
            "scale_sex_GJup_female": 1.1108,
            "scale_sex_calm_female": 1.2349,
        }

    else:
        raise ValueError(f"Unknown profile number: {profile_number}")


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
