import pandas as pd
import numpy as np
import random
from datetime import datetime
from datetime import timedelta
from datasets import load_dataset
from Download_datasets import Datsets
import re
from typing import Optional


# Grab datasets
# Add emmotion tags
# add time deltas
# clean up

datasets = Datsets()

def get_datasets(num_rows: int = 1, add_thoughts: bool = False):
    # Load all datasets; optionally augment with model-generated <think>
    return datasets.download_all(num_rows=num_rows, add_thoughts=add_thoughts)


if __name__ == "__main__":
    # Generate datasets with inserted model thoughts; surface errors directly
    data = get_datasets(num_rows=5, add_thoughts=True)
    for name, df in data.items():
        print(f"\n=== {name} ===")
        rows = len(df)
        has_target = isinstance(df, pd.DataFrame) and ('target' in df.columns)
        has_messages = isinstance(df, pd.DataFrame) and ('messages' in df.columns)
        tgt_cnt = int(df['target'].astype(str).str.contains('<think>', na=False).sum()) if has_target else 0
        msg_cnt = int(df['messages'].astype(str).str.contains('<think>', na=False).sum()) if has_messages else 0
        inserted_cnt = int(df['think_inserted'].fillna(False).astype(bool).sum()) if 'think_inserted' in df.columns else 0
        preexisting_msg_think = max(0, msg_cnt - inserted_cnt)
        print(f"rows={rows} | target_think={tgt_cnt} | messages_think={msg_cnt} | think_inserted={inserted_cnt} | think_preexisting_in_messages={preexisting_msg_think}")

        # Print explicit assistant <think> content (no silent catches)
        printed = 0
        if has_messages:
            import json, re
            for _, row in df.iterrows():
                msgs = row.get('messages')
                msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
                if isinstance(msgs, list):
                    for i, m in enumerate(msgs):
                        content = str(m.get('content',''))
                        if m.get('role') == 'assistant' and '<think>' in content:
                            think = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                            think_text = think.group(1).strip() if think else content
                            preview = think_text.replace('\n',' ')[:400]
                            origin = 'inserted' if bool(row.get('think_inserted', False)) else 'preexisting'
                            print(f"messages[{i}] assistant <think> ({origin}): {preview}")
                            printed += 1
                            break
                if printed >= 2:
                    break