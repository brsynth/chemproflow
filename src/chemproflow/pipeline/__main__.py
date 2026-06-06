import argparse
import logging
import os
import sys

import pandas as pd
from chemproflow.pipeline.model import CatalogTcid, ModelTcidInfer, ModelTransportInfer
from chemproflow.pipeline.report import Sample, Report
from chemproflow.pipeline.utils import prepare_df
from chemproflow import _version
from tqdm import tqdm


AP = argparse.ArgumentParser(description="")
AP_subparsers = AP.add_subparsers(help="Sub-commnands (use with -h for more info)")


def _cmd_pip(args):
    """Run ChemProFlow pipeline"""
    # Check arguments.
    logging.info("Start - Pipeline")

    if not args.input_smiles_str and not args.input_smiles_csv:
        logging.error("No input SMILES provided. Use --input-smiles-str or --input-smiles-csv")
        AP.exit(1)
    if args.input_smiles_csv and not os.path.isfile(args.input_smiles_csv):
        logging.error("CSV file does not exist: %s" % (args.input_smiles_csv,))
        AP.exit(1)
    if args.output_results_csv is None and args.output_results_html is None:
        logging.warning("No output files are indicated, results will not be saved")

    logging.info("Parse input")
    df = pd.DataFrame(columns=["smiles"])
    if args.input_smiles_str:
        df["smiles"] = args.input_smiles_str
    if args.input_smiles_csv:
        df_input = pd.read_csv(args.input_smiles_csv)
        if "SMILES" in df_input.columns:
            df_input.rename(columns={"SMILES": "smiles"}, inplace=True)
        if "smiles" not in df_input.columns:
            logging.warning("\"smiles\" column not found in the CSV header")
        else:
            df = pd.concat([df_input, df], ignore_index=True)
    df = df.dropna(subset=["smiles"])
    if df.empty:
        logging.error("No SMILES values found in the provided input")
        AP.exit(1)
    df = prepare_df(df=df)
    canonical_smiles = df["smiles_canonical"].tolist()

    # Check if input is in dataset - transport: columns = ['smiles', 'activity']
    model_transport = ModelTransportInfer(
        file_dataset_transport_csv=args.input_dataset_transport_csv,
        file_model_transport_pkl=args.input_model_transport_pkl,
        file_encoder_transport_pkl=args.input_encoder_transport_pkl,
        file_dirichlet_calibrator_pkl=args.input_dirichlet_calibrator_pkl,
    )
    df["dataset_transport"] = model_transport.in_dataset(smiles=canonical_smiles)
    df["pred_transport"] = model_transport.predict(smiles=canonical_smiles, batch_size=args.parameter_batch_size_int)

    # Run prediction - TCID
    model_tcid = ModelTcidInfer(
        file_dataset_tcid_csv=args.input_dataset_tcid_csv,
        file_model_tcid_pkl=args.input_model_tcid_pkl,
        file_encoder_tcid_pkl=args.input_encoder_tcid_pkl,
        file_threshold_tcid_json=args.input_threshold_tcid_json,
    )
    df["dataset_tcid"] = model_tcid.in_dataset(smiles=canonical_smiles)
    positive_transport = df["pred_transport"].astype(str).str.lower().isin(["positive", "true", "1", "yes", "y"])
    smiles = df.loc[positive_transport, 'smiles_canonical'].tolist()
    if len(smiles) > 0:
        tcids = model_tcid.predict(smiles=smiles, batch_size=args.parameter_batch_size_int)
        # Assign predicted TCIDs to dataframe
        for idx, pred_tcid in zip(df.index[positive_transport], tcids):
            df.at[idx, 'pred_tcid'] = pred_tcid

    # Catalog
    catalog_tcid = CatalogTcid(
        file_catalog_micro_organisms_csv=args.input_catalog_micro_organisms_csv,
        file_tcid_equivalent_json=args.input_tcid_equivalent_json,
    )
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Finding micro-organisms"):
        pred_tcids = row.get("pred_tcid") or []
        if pred_tcids:
            df.at[idx, 'accession_micro_organisms'] = catalog_tcid.map_tcids_organisms(tcids=pred_tcids)
    
    # Save results
    if args.output_results_csv:
        logging.info(f"Saving results to {args.output_results_csv}")
        df.to_csv(args.output_results_csv, index=False)
    if args.output_results_html:
        logging.info(f"Saving results to {args.output_results_html}")
        # Select template dir
        template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
        report = Report(template_path=os.path.join(template_dir, "report.html"))
        samples = []
        for record in df.to_dict(orient="records"):
            samples.append(Sample.from_dict(data=record))

        data = dict(app_version=_version.__version__, samples=samples)
        report.to_html(output_path=args.output_results_html, context=data)

    logging.info("End - Pipeline")
    return 0


P_pip = AP_subparsers.add_parser("pipeline", help=_cmd_pip.__doc__)
P_pip_inp = P_pip.add_argument_group("Input")
P_pip_inp.add_argument("--input-smiles-csv", help="Input file, with \"smiles\" column")
P_pip_inp.add_argument("--input-smiles-str", nargs="*", help="SMILES values separated by a space")
P_pip_inp.add_argument("--input-dataset-transport-csv", required=True, help="Path to transport dataset CSV")
P_pip_inp.add_argument("--input-dataset-tcid-csv", required=True, help="Path to TCID dataset CSV")
P_pip_inp.add_argument("--input-model-transport-pkl", required=True, help="Path to model checkpoint")
P_pip_inp.add_argument("--input-encoder-transport-pkl", required=True, help="Path to encoder pickle")
P_pip_inp.add_argument("--input-dirichlet-calibrator-pkl", required=True, help="Path to Dirichlet calibration pickle (defaults to dirichlet_calibrator.pkl next to the model)")
P_pip_inp.add_argument("--input-model-tcid-pkl", required=True, help="Path to model checkpoint")
P_pip_inp.add_argument("--input-encoder-tcid-pkl", required=True, help="Path to encoder pickle")
P_pip_inp.add_argument("--input-threshold-tcid-json", required=True, help="Path to threshold json")
P_pip_inp.add_argument("--input-catalog-micro-organisms-csv", required=True, help="Path to catalog of micro-organisms")
P_pip_inp.add_argument("--input-tcid-equivalent-json", required=True, help="Path input TCID file")
P_pip_par = P_pip.add_argument_group("Parameter")
P_pip_par.add_argument("--parameter-batch-size-int", type=int, default=16, help="Batch size")
P_pip_out = P_pip.add_argument_group("Output")
P_pip_out.add_argument("--output-results-csv", help="Path to output CSV file")
P_pip_out.add_argument("--output-results-html", help="Path to output HTML file")
P_pip.set_defaults(func=_cmd_pip)


# Help.
def print_help():
    """Display this program"s help"""
    AP.print_help()
    AP.exit()


# Version.
def print_version(_args):
    """Display this program"s version"""
    print(_version.__version__)


P_version = AP_subparsers.add_parser("version", help=print_version.__doc__)
P_version.set_defaults(func=print_version)


# Main.
def parse_args(args=None):
    """Parse the command line"""
    return AP.parse_args(args=args)


def main():
    """Entrypoint to commandline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M",
    )
    args = AP.parse_args()
    # No arguments or subcommands were given.
    if len(args.__dict__) < 1:
        print_help()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
