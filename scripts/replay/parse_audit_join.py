#!/usr/bin/env python3
"""
Parse ModSecurity audit JSON logs and join with existing parquet replay data
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_modsec_audit_json(audit_path: str) -> pd.DataFrame:
    """Parse ModSecurity audit logs in JSON format (one JSON per line)"""
    rows: List[Dict[str, Any]] = []
    
    with open(audit_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tx = obj.get("transaction", {})
                messages = tx.get("messages", [])
                
                # Extract request details
                request = tx.get("request", {})
                response = tx.get("response", {})
                uri = request.get("uri", "")
                method = request.get("method", "")
                http_code = response.get("http_code", 0)
                headers = request.get("headers", {}) or {}
                # Try common casings for our custom header
                replay_id: Optional[str] = None
                for k in ("X-Replay-ID", "X-Replay-Id", "x-replay-id"):
                    if k in headers:
                        replay_id = headers.get(k)
                        break
                
                # Collect rule IDs and anomaly scores
                rule_ids = []
                inbound_anomaly = 0
                outbound_anomaly = 0
                
                for msg in messages:
                    details = msg.get("details", {})
                    rule_id = details.get("ruleId")
                    if rule_id:
                        rule_ids.append(rule_id)
                    
                    # Check for anomaly scores in message
                    message_text = msg.get("message", "")
                    if "Inbound Anomaly Score" in message_text and "Total Score:" in message_text:
                        # Extract score from message like "Total Score: 5"
                        try:
                            score_part = message_text.split("Total Score:")[1].strip().split(")")[0]
                            inbound_anomaly = int(score_part)
                        except:
                            pass
                    elif "Outbound Anomaly Score" in message_text and "Total Score:" in message_text:
                        try:
                            score_part = message_text.split("Total Score:")[1].strip().split(")")[0]
                            outbound_anomaly = int(score_part)
                        except:
                            pass
                
                # Create row for this transaction
                if uri and uri != "/healthz":  # Skip healthcheck requests
                    row = {
                        "url": uri,
                        "method": method,
                        "status_code": http_code,
                        "rule_ids": ",".join(rule_ids) if rule_ids else None,
                        "inbound_anomaly": inbound_anomaly,
                        "outbound_anomaly": outbound_anomaly,
                        "blocked": 1 if http_code == 403 else 0,
                        "replay_id_audit": replay_id,
                        "timestamp": tx.get("time_stamp", ""),
                        "unique_id": tx.get("unique_id", "")
                    }
                    rows.append(row)
                    
            except Exception as e:
                print(f"Error parsing line {line_no}: {e}")
                continue
    
    return pd.DataFrame(rows)


def join_audit_with_parquet(audit_path: str, parquet_path: str, output_path: str):
    """Join audit logs with existing parquet replay data"""
    print(f"Parsing audit logs from: {audit_path}")
    audit_df = parse_modsec_audit_json(audit_path)
    
    print(f"Loading parquet data from: {parquet_path}")
    replay_df = pd.read_parquet(parquet_path)
    
    print(f"Audit logs: {len(audit_df)} entries")
    print(f"Replay data: {len(replay_df)} entries")

    # Ensure expected columns exist in replay
    for col in ["rule_ids", "inbound_anomaly", "outbound_anomaly"]:
        if col not in replay_df.columns:
            replay_df[col] = None

    # If we have replay-id in audit, perform a left-merge to enrich replay rows
    result_df: pd.DataFrame
    if len(audit_df) > 0 and "replay_id_audit" in audit_df.columns:
        # Deduplicate audit on replay_id (keep last occurrence)
        audit_dedup = audit_df.dropna(subset=["replay_id_audit"]).drop_duplicates(
            subset=["replay_id_audit"], keep="last"
        )[[
            "replay_id_audit", "rule_ids", "inbound_anomaly", "outbound_anomaly", "blocked"
        ]].rename(columns={
            "blocked": "blocked_audit"
        })
        result_df = replay_df.merge(
            audit_dedup,
            how="left",
            left_on="replay_id",
            right_on="replay_id_audit",
        )
        # Prefer audit blocked flag when available
        if "blocked_audit" in result_df.columns:
            result_df["blocked"] = result_df["blocked_audit"].fillna(result_df.get("blocked", 0)).astype("Int64")
            result_df = result_df.drop(columns=[c for c in ["replay_id_audit", "blocked_audit"] if c in result_df.columns])
    else:
        # Fallback: no linkable audit; return replay with empty enrichment
        print("No linkable audit data found (missing replay_id); using original replay data")
        result_df = replay_df.copy()
    
    print(f"Saving joined data to: {output_path}")
    result_df.to_parquet(output_path, index=False)
    
    print(f"Final dataset: {len(result_df)} entries")
    return result_df


def print_metrics(df: pd.DataFrame):
    print("\n=== Metrics ===")
    if "blocked" in df.columns and df["blocked"].notna().any():
        total = len(df)
        blocked = int(df["blocked"].fillna(0).astype(int).sum())
        bypass = total - blocked
        bypass_ratio = (bypass / total) if total > 0 else 0.0
        print(f"Total: {total}  Blocked: {blocked}  Bypass: {bypass}  Bypass ratio: {bypass_ratio:.3f}")
    if "resp_ms" in df.columns and df["resp_ms"].notna().any():
        p50 = float(df["resp_ms"].quantile(0.5))
        p95 = float(df["resp_ms"].quantile(0.95))
        print(f"Latency p50: {p50:.2f} ms  p95: {p95:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, help="Path to ModSecurity audit log file")
    parser.add_argument("--in_parquet", required=True, help="Input parquet file to join with")
    parser.add_argument("--out_parquet", required=True, help="Output parquet file")
    
    args = parser.parse_args()
    
    result_df = join_audit_with_parquet(args.audit, args.in_parquet, args.out_parquet)
    
    # Show summary
    print("\n=== Summary ===")
    print(f"Total rows: {len(result_df)}")
    if 'blocked' in result_df.columns:
        blocked_count = result_df['blocked'].fillna(0).astype(int).sum()
        print(f"Blocked requests: {blocked_count}")
    if 'rule_ids' in result_df.columns:
        has_rules = result_df['rule_ids'].notna().sum()
        print(f"Entries with rule_ids: {has_rules}")
    if 'inbound_anomaly' in result_df.columns:
        avg_anomaly = result_df['inbound_anomaly'].mean() if result_df['inbound_anomaly'].notna().any() else 0
        print(f"Average inbound anomaly: {avg_anomaly:.2f}")
    print_metrics(result_df)