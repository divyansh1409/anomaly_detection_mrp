import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_openstack_features():
    print("Extracting features for OpenStack dataset...")
    df = pd.read_csv('OpenStack_2k.log_structured.csv')
    print(f"Loaded {len(df)} OpenStack log entries")
    
    features = {}
    
    # 1. Basic Text Features
    features['log_length'] = df['Content'].str.len()
    features['word_count'] = df['Content'].str.split().str.len()
    features['char_count_no_spaces'] = df['Content'].str.replace(' ', '').str.len()
    
    # 2. OpenStack-Specific Patterns (based on actual dataset analysis)
    features['contains_nova'] = df['Content'].str.contains(r'\bnova\b', case=False).astype(int)
    features['contains_http'] = df['Content'].str.contains(r'\bHTTP/\d+\.\d+\b', case=False).astype(int)
    features['contains_get'] = df['Content'].str.contains(r'\bGET\b', case=False).astype(int)
    features['contains_post'] = df['Content'].str.contains(r'\bPOST\b', case=False).astype(int)
    features['contains_put'] = df['Content'].str.contains(r'\bPUT\b', case=False).astype(int)
    features['contains_delete'] = df['Content'].str.contains(r'\bDELETE\b', case=False).astype(int)
    features['contains_status'] = df['Content'].str.contains(r'\bstatus:\s*\d+\b', case=False).astype(int)
    features['contains_len'] = df['Content'].str.contains(r'\blen:\s*\d+\b', case=False).astype(int)
    features['contains_time'] = df['Content'].str.contains(r'\btime:\s*[\d\.]+\b', case=False).astype(int)
    features['contains_servers'] = df['Content'].str.contains(r'\bservers\b', case=False).astype(int)
    features['contains_detail'] = df['Content'].str.contains(r'\bdetail\b', case=False).astype(int)
    features['contains_v2'] = df['Content'].str.contains(r'\bv2\b', case=False).astype(int)
    
    # 3. OpenStack-Specific Error Patterns
    features['contains_error'] = df['Content'].str.contains(r'\b(error|Error|ERROR)\b', case=False).astype(int)
    features['contains_failed'] = df['Content'].str.contains(r'\b(failed|Failed)\b', case=False).astype(int)
    features['contains_exception'] = df['Content'].str.contains(r'\b(exception|Exception)\b', case=False).astype(int)
    features['contains_timeout'] = df['Content'].str.contains(r'\b(timeout|Timeout)\b', case=False).astype(int)
    features['contains_warning'] = df['Content'].str.contains(r'\b(warning|Warning|WARNING)\b', case=False).astype(int)
    
    # 4. OpenStack-Specific System Patterns
    features['contains_ip_address'] = df['Content'].str.contains(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', case=False).astype(int)
    features['contains_port'] = df['Content'].str.contains(r':\d{4,5}\b', case=False).astype(int)
    features['contains_uuid'] = df['Content'].str.contains(r'[a-f0-9]{32}', case=False).astype(int)
    features['contains_compute'] = df['Content'].str.contains(r'\bcompute\b', case=False).astype(int)
    features['contains_api'] = df['Content'].str.contains(r'\bapi\b', case=False).astype(int)
    features['contains_scheduler'] = df['Content'].str.contains(r'\bscheduler\b', case=False).astype(int)
    features['contains_metadata'] = df['Content'].str.contains(r'\bmetadata\b', case=False).astype(int)
    
    # 5. Numeric Patterns (specific to OpenStack)
    features['ip_address_count'] = df['Content'].str.count(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    features['port_number_count'] = df['Content'].str.count(r':\d{4,5}\b')
    features['status_code_count'] = df['Content'].str.count(r'status:\s*\d+')
    features['response_time_count'] = df['Content'].str.count(r'time:\s*[\d\.]+')
    features['response_length_count'] = df['Content'].str.count(r'len:\s*\d+')
    features['uuid_count'] = df['Content'].str.count(r'[a-f0-9]{32}')
    
    # 6. Event Template Analysis
    features['event_template_length'] = df['EventTemplate'].str.len()
    features['event_template_word_count'] = df['EventTemplate'].str.split().str.len()
    features['has_parameters'] = (df['EventTemplate'].str.contains(r'<\*>')).astype(int)
    features['parameter_count'] = df['EventTemplate'].str.count(r'<\*>')
    
    # 7. Component Analysis
    features['unique_components'] = df['Component'].nunique()
    
    # 8. Temporal Features
    if 'Timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
        features['hour_of_day'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_business_hours'] = ((features['hour_of_day'] >= 9) & (features['hour_of_day'] <= 17)).astype(int)
    
    # 9. Sequential Features
    features['line_id'] = df['LineId']
    
    # 10. Label Information (for supervised learning)
    if 'Label' in df.columns:
        features['is_anomaly'] = (df['Label'] != '-').astype(int)
    
    # Create features DataFrame
    feature_df = pd.DataFrame(features)
    
    # Fill NaN values with appropriate defaults
    feature_df = feature_df.fillna(0)
    
    # Additional cleanup: replace any remaining empty strings with 0 for numeric columns
    for col in feature_df.columns:
        if col != 'is_anomaly' and feature_df[col].dtype in ['int64', 'float64']:
            feature_df[col] = feature_df[col].replace('', 0)
            feature_df[col] = feature_df[col].replace('nan', 0)
            feature_df[col] = feature_df[col].replace('NaN', 0)
    
    # Add component-based features (only the ones that exist in OpenStack)
    component_dummies = pd.get_dummies(df['Component'], prefix='component')
    feature_df = pd.concat([feature_df, component_dummies], axis=1)
    
    # Add event template-based features (limit to most common events to avoid sparse columns)
    event_counts = df['EventId'].value_counts()
    common_events = event_counts[event_counts >= 5].index.tolist()
    if len(common_events) > 0:
        event_dummies = pd.get_dummies(df['EventId'][df['EventId'].isin(common_events)], prefix='event')
        # Fill any NaN values in event dummies with 0
        event_dummies = event_dummies.fillna(0)
        feature_df = pd.concat([feature_df, event_dummies], axis=1)
    
    # Add severity level features
    level_dummies = pd.get_dummies(df['Level'], prefix='level')
    feature_df = pd.concat([feature_df, level_dummies], axis=1)
    
    # Remove columns that are entirely zero or empty (useless features)
    zero_cols = []
    empty_cols = []
    for col in feature_df.columns:
        if feature_df[col].dtype in ['int64', 'float64']:
            # Check for all zeros
            if feature_df[col].sum() == 0:
                zero_cols.append(col)
            # Check for all NaN or empty strings
            elif feature_df[col].isna().all() or (feature_df[col].astype(str) == '').all():
                empty_cols.append(col)
        else:
            # For non-numeric columns, check for all empty strings
            if (feature_df[col] == '').all():
                empty_cols.append(col)
    
    cols_to_remove = zero_cols + empty_cols
    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} useless columns:")
        if zero_cols:
            print(f"  {len(zero_cols)} columns with all zeros: {zero_cols[:5]}")
        if empty_cols:
            print(f"  {len(empty_cols)} columns with all empty values: {empty_cols[:5]}")
        feature_df = feature_df.drop(columns=cols_to_remove)

    # FINAL: Fill any remaining NaN values with 0
    feature_df = feature_df.fillna(0)
    
    # Convert all boolean columns to integers (0/1)
    for col in feature_df.columns:
        if feature_df[col].dtype == 'bool':
            feature_df[col] = feature_df[col].astype(int)
        elif feature_df[col].dtype == 'object':
            # Check if column contains boolean strings or mixed boolean/numeric
            unique_vals = set(feature_df[col].unique())
            if unique_vals.issubset({'True', 'False', '0', '1'}):
                # Convert to numeric: 'True'/'1' -> 1, 'False'/'0' -> 0
                feature_df[col] = feature_df[col].replace({'True': 1, 'False': 0, '1': 1, '0': 0}).astype(int)
    
    # Force all event, component, and level columns to int
    for col in feature_df.columns:
        if col.startswith('event_') or col.startswith('component_') or col.startswith('level_'):
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0).astype(int)
    
    # Save features
    feature_df.to_csv('OpenStack_features.csv', index=False)
    print(f"Extracted {len(feature_df.columns)} features for OpenStack dataset")
    print(f"Features saved to OpenStack_features.csv")
    
    # Print feature summary
    print("\nFeature Summary:")
    print(f"Total samples: {len(feature_df)}")
    print(f"Total features: {len(feature_df.columns)}")
    print(f"Anomaly rate: {feature_df['is_anomaly'].mean():.2%}" if 'is_anomaly' in feature_df.columns else "No labels available")
    
    # Print feature statistics for binary features
    binary_features = []
    for col in feature_df.columns:
        if feature_df[col].dtype in ['int64', 'float64']:
            unique_vals = feature_df[col].unique()
            if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                binary_features.append(col)
    
    print(f"\nBinary features with non-zero values:")
    for col in binary_features:
        if col != 'is_anomaly' and feature_df[col].sum() > 0:
            print(f"  {col}: {feature_df[col].sum()} occurrences ({feature_df[col].mean():.2%})")
    
    return feature_df

if __name__ == "__main__":
    openstack_features = extract_openstack_features() 