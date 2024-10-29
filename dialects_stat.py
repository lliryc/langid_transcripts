import os
import glob
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import json

# Initialize a Counter to hold label counts
label_counter = Counter()

# Get list of all .tsv files in the 'transcripts_labels' folder matching '@*.tsv'
files = glob.glob('smb_share/transcripts_labels/@*.tsv')

# Loop over each file
for filename in files:
    print(f"Processing file: {filename}")
    # Read the tsv file into a pandas DataFrame
    df = pd.read_csv(filename, sep='\t')
    
    # Check if 'langid' column exists
    if 'langid' not in df.columns:
        print(f"'langid' column not found in file {filename}. Skipping.")
        continue
    
    # Loop over each value in the 'langid' column
    for idx, langid_str in enumerate(df['langid']):
        # Parse the string as a dictionary
        try:
            s = langid_str.strip('"')
            s = s.replace('""', '"')
            data = json.loads(s)
            # Extract the keys (labels)
            labels = data.keys()
            # filter out the labels that don't contain '_Arab' substring
            labels = [label for label in labels if '_Arab' in label]
            # Update the Counter with the labels
            label_counter.update(labels)
        except json.JSONDecodeError as e:
            print(f"Error parsing langid string at index {idx} in file {filename}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error at index {idx} in file {filename}: {e}")
            continue

# After processing all files, plot the counts
labels = list(label_counter.keys())
print('Overall count of dialects:', len(labels))

counts = list(label_counter.values())

# Plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(labels, counts, color='skyblue')

# Set y-axis to logarithmic scale
plt.yscale('log')

# Annotate each bar with its count
for i, count in enumerate(counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.xlabel('Dialects')
plt.ylabel('Counts')
plt.title('Counts of Labels in langid Column')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()
