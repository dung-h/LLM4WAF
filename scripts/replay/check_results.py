import pandas as pd

df = pd.read_parquet("artifacts/replays/sqli_qwen2_20_auth_final.joined.parquet")

print("=== Sample Payloads and Results ===\n")
print(df[["payload", "status_code", "blocked", "resp_len", "valid"]].head(10).to_string())

print("\n=== Full Metrics ===")
print(f"Total: {len(df)}")
print(f"Valid: {df['valid'].sum()}")
print(f"Blocked (403): {(df['status_code'] == 403).sum()}")
print(f"Status 200: {(df['status_code'] == 200).sum()}")
print(f"Avg response length: {df['resp_len'].mean():.0f} bytes")

# Check if responses contain SQL data
print("\n=== Checking Actual Content ===")
if 'resp_len' in df.columns:
    print(f"Min resp_len: {df['resp_len'].min()}")
    print(f"Max resp_len: {df['resp_len'].max()}")
    print(f"Median resp_len: {df['resp_len'].median()}")
