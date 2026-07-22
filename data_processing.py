import os
import argparse
import time
import numpy as np
import pandas as pd

from chronotrace.adapters import EventLogCSVAdapter
from chronotrace.models import TraceDataset
from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor

parser = argparse.ArgumentParser(
    description="Process Transformer - Data Processing.")

parser.add_argument("--dataset", 
    type=str, 
    default="helpdesk", 
    help="dataset name")

parser.add_argument("--dir_path", 
    type=str, 
    default="./datasets", 
    help="path to store processed data")

parser.add_argument("--raw_log_file", 
    type=str, 
    default="./datasets/raw/helpdesk/finale.csv", 
    help="path to raw csv log file")

parser.add_argument("--new_dataset",
    type=str,
    default=None,
    help="path to ChronoTrace new_dataset.json")
parser.add_argument("--train_log_file",
    type=str,
    default=None,
    help="path to pre-split train csv file")
parser.add_argument("--test_log_file",
    type=str,
    default=None,
    help="path to pre-split test csv file")

parser.add_argument("--task", 
    type=constants.Task, 
    default=constants.Task.REMAINING_TIME, 
    help="task name")

parser.add_argument("--sort_temporally", 
    type=bool, 
    default=False, 
    help="sort cases by timestamp")

parser.add_argument("--columns",
    type=str,
    default="helpdesk",
    help="column preset: 'helpdesk' = [Case ID, Activity, Complete Timestamp]; "
         "'xes' = [case:concept:name, concept:name, time:timestamp]; "
         "or a comma-separated triplet 'case_col,activity_col,time_col'")

args = parser.parse_args()

if args.columns == "helpdesk":
    _columns = ["Case ID", "Activity", "Complete Timestamp"]
elif args.columns == "xes":
    _columns = ["case:concept:name", "concept:name", "time:timestamp"]
else:
    _columns = [c.strip() for c in args.columns.split(",")]
    assert len(_columns) == 3, "--columns needs exactly 3 values: case, activity, time"

trace_dataset = None
train_cases = None
test_cases = None
if args.train_log_file and args.test_log_file:
    train_adapter = EventLogCSVAdapter(args.train_log_file)
    train_ds = (train_adapter.load_new_dataset(args.new_dataset)
                if args.new_dataset else train_adapter.load(train_adapter.csv_path.parent.name))
    test_adapter = EventLogCSVAdapter(args.test_log_file)
    test_ds = test_adapter.load(test_adapter.csv_path.parent.name)
    trace_dataset = TraceDataset(
        dataset_name=train_ds.dataset_name, workload_id=train_ds.workload_id,
        workload_description=train_ds.workload_description,
        traces=train_ds.traces + test_ds.traces, meta=dict(train_ds.meta))
    train_cases = [t.trace_id for t in train_ds.traces]
    test_cases = [t.trace_id for t in test_ds.traces]
elif args.train_log_file or args.test_log_file:
    raise ValueError("--train_log_file and --test_log_file must be given together")
elif args.new_dataset:
    raise ValueError("--new_dataset requires --train_log_file and --test_log_file (augmentation is train-only)")

if __name__ == "__main__": 
    # Process raw logs
    start = time.time()
    data_processor = LogsDataProcessor(name=args.dataset, 
        filepath=args.raw_log_file, 
        columns=_columns,
        dir_path=args.dir_path, pool=1,
        trace_dataset=trace_dataset)
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally)
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally,
        train_cases=train_cases, test_cases=test_cases)
    end = time.time()
    print(f"Total processing time: {end - start}")
