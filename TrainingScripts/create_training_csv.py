import csv
import os
import glob
import time
import json
import random
from TrainingScripts.ExtraTools import TimeTools, EmotionEngine

def read_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8', errors='ignore') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def parse_data(data):
    parsed_data = []
    for row in data:
        # Columns: Speaker, Message Content, Date And Time, Attachments
        if len(row) < 3:
            continue
        parsed_row = {
            'Speaker': row[0],
            'Content': row[1],
            'Date And Time': row[2],
            'Attachments?': row[3] if len(row) > 3 else ''
        }
        parsed_data.append(parsed_row)
    return parsed_data

def section_data_by_time(data, hours=6):
    """
    Aggregates messages into sections based on a fixed time window.
    All messages within the window are grouped together.
    """
    from datetime import datetime, timedelta
    import re

    if not data or len(data) == 0:
        return []
    
    # Remove header if present
    if data[0].get('Speaker') == 'Title':
        data = data[1:]
    
    sections = []
    current_section = []
    section_start_time = None
    
    for row in data:
        # Skip empty messages
        if not row.get('Date And Time') or row['Date And Time'].strip() == '':
            continue
            
        try:
            # Parse date - format example: "Friday, Feb 5, 2021 8:51 PM"
            date_str = row['Date And Time'].strip()
            # Remove day of week if present
            date_str = re.sub(r'^[A-Za-z]+,\s*', '', date_str)
            current_time = datetime.strptime(date_str, "%b %d, %Y %I:%M %p")
            
            # Start a new section if this is the first message or if we've exceeded the time window
            if section_start_time is None:
                section_start_time = current_time
                current_section = [row]
            elif (current_time - section_start_time) <= timedelta(hours=hours):
                current_section.append(row)
            else:
                # Save current section and start a new one
                if current_section:
                    sections.append(current_section)
                section_start_time = current_time
                current_section = [row]
        except Exception as e:
            # If date parsing fails, add to current section anyway
            if current_section is not None:
                current_section.append(row)
            else:
                current_section = [row]
    
    # Don't forget the last section
    if current_section:
        sections.append(current_section)
    
    return sections

def make_time_tag(ts_str):
    dt = TimeTools.parse_timestamp(ts_str)
    if not dt:
        return ""
    hour = dt.hour
    if 0 <= hour < 6:
        tod = "Late night"
    elif 6 <= hour < 12:
        tod = "Morning"
    elif 12 <= hour < 15:
        tod = "Afternoon"
    elif 15 <= hour < 18:
        tod = "Late day"
    elif 18 <= hour < 21:
        tod = "Early evening"
    else:
        tod = "Late evening"
    return f"[TIME:{tod}]"

def build_turn_rows(section_rows):
    """Build turn-by-turn progressive examples for a conversation section (5-hour chunks).
    Each row corresponds to a 'Me' turn; prior turns are context with tags.
    """
    rows = [r for r in section_rows if r.get('Speaker') not in ('Title', 'Conversation between Me and') and (r.get('Content') or '').strip()]
    if not rows:
        return []
    entries = []
    for k in range(len(rows)):
        if rows[k].get('Speaker') != 'Me':
            continue
        messages = []
        next_ts = (rows[k].get('Date And Time') or '').strip()
        next_dt = TimeTools.parse_timestamp(next_ts)
        for j in range(0, k):
            sp = rows[j].get('Speaker')
            content = (rows[j].get('Content') or '').strip()
            ts = (rows[j].get('Date And Time') or '').strip()
            ts_tag = f"[TS:{ts}]" if ts else ""
            time_tag = make_time_tag(ts)
            if sp == 'Me':
                msg = {'role': 'assistant', 'content': f"{ts_tag} {time_tag} {content}".strip()}
            else:
                prev_dt = TimeTools.parse_timestamp(ts)
                delta_tag = TimeTools.make_delta_tag(TimeTools.delta_seconds(prev_dt, next_dt))
                emo_tag = EmotionEngine.tag(content)
                msg = {'role': 'user', 'content': f"{delta_tag} {ts_tag} {time_tag} {emo_tag} {content}".strip()}
            messages.append(msg)
        target_ts = (rows[k].get('Date And Time') or '').strip()
        target_time_tag = make_time_tag(target_ts)
        target = f"[TS:{target_ts}] {target_time_tag} {(rows[k].get('Content') or '').strip()}".strip()
        entries.append({'messages': messages, 'target': target})
    return entries

# fix_data removed: logic replaced by block grouping in build_block_pairs
        
            

def build_progressive_conversations(conversations):
    """Pass-through: conversations are already progressive entries from block pairs."""
    progressive_data = []
    for conv in conversations:
        progressive_data.extend(conv)
    return progressive_data

# Get all CSV files in the Converted_CSVs directory
csv_files = glob.glob("Converted_CSVs/*.csv")

print(f"Found {len(csv_files)} CSV files to process\n")

# Process all CSV files and aggregate training data
total_data = []
start_time = time.time()
last_update = start_time

for idx, file_name in enumerate(csv_files, start=1):
    now = time.time()
    if now - last_update > 1.0:
        elapsed = now - start_time
        avg_per_file = elapsed / max(1, (idx - 1))
        remaining_files = len(csv_files) - (idx - 1)
        eta_sec = avg_per_file * remaining_files
        print(f"Progress: {idx-1}/{len(csv_files)} files | Elapsed: {int(elapsed)}s | ETA: {int(eta_sec)}s")
        last_update = now
    print(f"Processing: {file_name}")
    
    try:
        data = read_csv(file_name)
        parsed_data = parse_data(data)
        
        # Section data into 5-hour conversation blocks
        sections = section_data_by_time(parsed_data, hours=5)
        
        # Build turn-by-turn entries per section and aggregate
        conversations = []
        for section in sections:
            entries = build_turn_rows(section)
            if entries:
                conversations.append(entries)
        # Flatten to training examples
        file_training_data = build_progressive_conversations(conversations)
        total_data.extend(file_training_data)
        
        print(f"  Added {len(file_training_data)} training examples from this file")
        
    except Exception as e:
        print(f"  Error processing {file_name}: {e}")
    
    print()

print(f"\n{'='*60}")
print(f"TOTAL TRAINING EXAMPLES: {len(total_data)}")
print(f"{'='*60}\n")
total_elapsed = time.time() - start_time
print(f"Completed in {int(total_elapsed)} seconds (~{int(total_elapsed/60)} min)")

# Print a few sample examples
for i in range(min(3, len(total_data))):
    print(f"Sample Example {i+1}:")
    print(f"  Context messages: {len(total_data[i]['messages'])}")
    for msg in total_data[i]['messages']:
        print(f"    {msg['role']}: {msg['content'][:50]}...")
    print(f"  Target: {total_data[i]['target'][:50]}...")
    print()

# Shuffle the data for random train/test split
random.seed(42)  # For reproducibility
random.shuffle(total_data)

# Split into 80% train, 20% test
split_index = int(len(total_data) * 0.8)
train_data = total_data[:split_index]
test_data = total_data[split_index:]

print(f"\nTrain examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}\n")

# Export train data to CSV
def export_to_csv(data, filename):
    """Export training data to CSV with messages serialized as JSON"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['messages', 'target'])
        
        # Write data rows
        for entry in data:
            # Serialize messages list to JSON string
            messages_json = json.dumps(entry['messages'])
            writer.writerow([messages_json, entry['target']])
    
    print(f"Exported {len(data)} examples to {filename}")

# Export both train and test datasets
os.makedirs('TrainingData', exist_ok=True)
export_to_csv(train_data, os.path.join('TrainingData','training_data.csv'))
export_to_csv(test_data, os.path.join('TrainingData','test_data.csv'))

print("\n✓ Export complete!")

