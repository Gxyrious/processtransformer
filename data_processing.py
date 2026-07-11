import os
import argparse
import time
import numpy as np
import pandas as pd

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

if __name__ == "__main__": 
    # Process raw logs
    start = time.time()
    data_processor = LogsDataProcessor(name=args.dataset, 
        filepath=args.raw_log_file, 
        columns=_columns,
        dir_path=args.dir_path, pool = 1) #changed from 4 to 1
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally)
    end = time.time()
    print(f"Total processing time: {end - start}")
