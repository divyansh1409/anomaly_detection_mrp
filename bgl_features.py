import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_bgl_features():
    print("Extracting features for BGL dataset...")
    df = pd.read_csv('BGL_2k.log_structured.csv')
    print(f"Loaded {len(df)} BGL log entries")
    
    features = {}
    
    # 1. Basic Text Features
    features['log_length'] = df['Content'].str.len()
    features['word_count'] = df['Content'].str.split().str.len()
    features['char_count_no_spaces'] = df['Content'].str.replace(' ', '').str.len()
    
    # 2. BGL-Specific Hardware Patterns (based on actual dataset analysis)
    features['contains_cache'] = df['Content'].str.contains(r'\b(cache|Cache)\b', case=False).astype(int)
    features['contains_core'] = df['Content'].str.contains(r'\b(core|Core)\b', case=False).astype(int)
    features['contains_instruction'] = df['Content'].str.contains(r'\b(instruction|Instruction)\b', case=False).astype(int)
    features['contains_parity'] = df['Content'].str.contains(r'\b(parity|Parity)\b', case=False).astype(int)
    features['contains_exception'] = df['Content'].str.contains(r'\b(exception|Exception)\b', case=False).astype(int)
    features['contains_alignment'] = df['Content'].str.contains(r'\b(alignment|Alignment)\b', case=False).astype(int)
    features['contains_double_hummer'] = df['Content'].str.contains(r'\b(double-hummer|Double-hummer)\b', case=False).astype(int)
    
    # 3. BGL-Specific Error Patterns (only those that actually appear)
    features['contains_ce_sym'] = df['Content'].str.contains(r'\bCE sym\b', case=False).astype(int)
    features['contains_failed'] = df['Content'].str.contains(r'\b(failed|Failed)\b', case=False).astype(int)
    features['contains_error'] = df['Content'].str.contains(r'\b(error|Error|ERROR)\b', case=False).astype(int)
    # Removed contains_fatal as it has all zeros
    features['contains_interrupt'] = df['Content'].str.contains(r'\b(interrupt|Interrupt)\b', case=False).astype(int)
    features['contains_tlb'] = df['Content'].str.contains(r'\b(TLB|tlb)\b', case=False).astype(int)
    features['contains_storage'] = df['Content'].str.contains(r'\b(storage|Storage)\b', case=False).astype(int)
    
    # 4. BGL-Specific System Patterns (only those that actually appear)
    features['contains_ciod'] = df['Content'].str.contains(r'\b(ciod|Ciod)\b', case=False).astype(int)
    features['contains_socket'] = df['Content'].str.contains(r'\b(socket|Socket)\b', case=False).astype(int)
    features['contains_message'] = df['Content'].str.contains(r'\b(message|Message)\b', case=False).astype(int)
    features['contains_directory'] = df['Content'].str.contains(r'\b(directory|Directory)\b', case=False).astype(int)
    features['contains_chdir'] = df['Content'].str.contains(r'\b(chdir|Chdir)\b', case=False).astype(int)
    
    # 5. Numeric Patterns (specific to BGL)
    features['hex_address_count'] = df['Content'].str.count(r'0x[0-9a-fA-F]+')
    features['ip_address_count'] = df['Content'].str.count(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    features['port_number_count'] = df['Content'].str.count(r':\d{4,5}\b')
    features['core_number_count'] = df['Content'].str.count(r'core\.\d+')
    features['mask_value_count'] = df['Content'].str.count(r'mask 0x[0-9a-fA-F]+')
    
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
    
    feature_df = feature_df.fillna(0)  # Fill numeric NaN values with 0
    
    for col in feature_df.columns:
        if col != 'is_anomaly' and feature_df[col].dtype in ['int64', 'float64']:
            feature_df[col] = feature_df[col].replace('', 0)
            feature_df[col] = feature_df[col].replace('nan', 0)
            feature_df[col] = feature_df[col].replace('NaN', 0)
    
    # Add component-based features 
    component_dummies = pd.get_dummies(df['Component'], prefix='component')
    feature_df = pd.concat([feature_df, component_dummies], axis=1)
    
    # Add event template-based features 
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

    #Fill any remaining NaN values with 0
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
    feature_df.to_csv('BGL_features.csv', index=False)
    print(f"Extracted {len(feature_df.columns)} features for BGL dataset")
    print(f"Features saved to BGL_features.csv")
    
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
    bgl_features = extract_bgl_features() 