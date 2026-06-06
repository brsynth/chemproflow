import pandas as pd
from chemproflow.utils.molecule import fmt_smiles


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df_col = pd.DataFrame(columns=["dataset_transport", "dataset_tcid", "pred_transport", "pred_tcid", "id_micro_organisms", "name_micro_organisms"])
    df = pd.concat([df, df_col], axis=1)
    df["pred_tcid"] = [[] for _ in range(len(df))]
    df["accession_micro_organisms"] = [[] for _ in range(len(df))]
    df["smiles_canonical"] = df["smiles"].apply(fmt_smiles)
    return df