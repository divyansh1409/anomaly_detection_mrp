import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re

# Dataset paths
paths = {
    "HDFS": "HDFS_2k.log_structured.csv",
    "BGL": "BGL_2k.log_structured.csv",
    "OpenStack": "OpenStack_2k.log_structured.csv",
    "Thunderbird": "Thunderbird_2k.log_structured.csv",
    "Zookeeper": "Zookeeper_2k.log_structured.csv"
}

# Load datasets
datasets = {name: pd.read_csv(path) for name, path in paths.items()}
for df in datasets.values():
    df.dropna(axis=1, how='all', inplace=True)

def generate_common_plots(df, name, has_level=True, has_numeric=False, numeric_event_id=None, numeric_pattern=None):
    # Component Frequency
    plt.figure(figsize=(10, 6))
    component_counts = df['Component'].value_counts().head(10)
    sns.barplot(x=component_counts.values, y=component_counts.index, palette='viridis')
    plt.title(f"Top Logged Components in {name} Logs")
    plt.xlabel("Number of Log Entries")
    plt.ylabel("Component")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_component_frequency.png")
    plt.close()

    # Event Template Distribution with percentage annotations
    plt.figure(figsize=(10, 6))
    top_event_counts = df['EventId'].value_counts().head(10)
    top_event_percents = top_event_counts / df['EventId'].count() * 100
    sns.barplot(x=top_event_counts.values, y=top_event_counts.index, palette='plasma')
    for i, val in enumerate(top_event_counts.values):
        percent = top_event_percents.values[i]
        plt.text(val + 5, i, f"{percent:.1f}%", va='center')
    plt.title(f"Top 10 Event Templates in {name} Logs")
    plt.xlabel("Number of Log Entries")
    plt.ylabel("Event ID")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_event_distribution.png")
    plt.close()

    # Log Level Distribution
    if has_level and 'Level' in df.columns:
        plt.figure(figsize=(6, 4))
        level_counts = df['Level'].value_counts()
        sns.barplot(x=level_counts.values, y=level_counts.index, palette='coolwarm')
        plt.title(f"Log Severity Level Distribution in {name}")
        plt.xlabel("Number of Log Entries")
        plt.ylabel("Log Level")
        plt.tight_layout()
        plt.savefig(f"{name.lower()}_log_level_distribution.png")
        plt.close()

    # Word Cloud
    text = ' '.join(df['EventTemplate'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud of {name} Event Templates")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_wordcloud.png")
    plt.close()

    # Numeric Value Distribution (Optional)
    if has_numeric and numeric_event_id and numeric_pattern:
        subset_logs = df[df['EventId'] == numeric_event_id]
        numeric_values = subset_logs['Content'].str.extract(numeric_pattern)[0].dropna().astype(float)
        if not numeric_values.empty:
            plt.figure(figsize=(10, 5))
            sns.histplot(numeric_values, bins=30, kde=True, color='skyblue')
            plt.title(f"Value Distribution for Event ID {numeric_event_id} ({name})")
            plt.xlabel("Extracted Numeric Value")
            plt.ylabel("Frequency")
            plt.ticklabel_format(style='plain', axis='x')  # Disable scientific notation
            plt.tight_layout()
            plt.savefig(f"{name.lower()}_{numeric_event_id.lower()}_value_distribution.png")
            plt.close()

# Generate plots
generate_common_plots(datasets["HDFS"], "HDFS", has_numeric=True, numeric_event_id="E6", numeric_pattern=r"blk_(-?\d+)")
generate_common_plots(datasets["BGL"], "BGL", has_numeric=True, numeric_event_id="E67", numeric_pattern=r"(-?\d+)")
generate_common_plots(datasets["OpenStack"], "OpenStack", has_numeric=False)
generate_common_plots(datasets["Thunderbird"], "Thunderbird", has_level=False, has_numeric=False)
generate_common_plots(datasets["Zookeeper"], "Zookeeper", has_numeric=False)
