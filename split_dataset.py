import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description="Process Transformer - split raw log into train/test by case.")
parser.add_argument("--raw_log_file", 
    type=str, 
    required=True,
    help="path to raw csv log file")
parser.add_argument("--output_dir", 
    type=str, 
    required=True,
    help="path to output directory")
parser.add_argument("--case_column", 
    type=str, 
    default="Case ID",
    help="case id column name")
parser.add_argument("--train_ratio", 
    type=float, 
    default=0.80,
    help="train split ratio")
args = parser.parse_args()

df = pd.read_csv(args.raw_log_file)
cases = list(df[args.case_column].unique())
n_train = int(len(cases) * args.train_ratio)
train_cases = set(cases[:n_train])
train_df = df[df[args.case_column].isin(train_cases)]
test_df = df[~df[args.case_column].isin(train_cases)]
train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
test_df.to_csv(f"{args.output_dir}/test.csv", index=False)
print(f"train: {n_train} cases, {len(train_df)} rows -> {args.output_dir}/train.csv")
print(f"test: {len(cases) - n_train} cases, {len(test_df)} rows -> {args.output_dir}/test.csv")
