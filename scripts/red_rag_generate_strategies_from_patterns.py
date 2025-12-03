import json
import os
import sys
from collections import defaultdict
from datetime import datetime
import argparse

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def generate_strategies(input_pattern_file, output_strategies_file):
    print(f"Generating strategies from patterns in {input_pattern_file}...")
    
    patterns = []
    if os.path.exists(input_pattern_file):
        with open(input_pattern_file, 'r', encoding='utf-8') as f:
            for line in f:
                patterns.append(json.loads(line.strip()))

    # Group patterns by heuristic "family"
    strategy_groups = defaultdict(lambda: {"patterns": [], "attack_type": "UNKNOWN", "idea": ""})

    for p in patterns:
        attack_type = p.get("attack_type", "UNKNOWN")
        technique = p.get("technique", "").lower()
        
        strategy_family = None
        idea_desc = ""

        # SQLI Strategies
        if attack_type == "SQLI":
            if "double url" in technique or "triple url" in technique or "url encode" in technique or "multi-encoding" in technique:
                strategy_family = "multi_url_encoding_sqli"
                idea_desc = "Sử dụng nhiều lớp URL encoding để vượt qua các bộ lọc WAF nhận diện chuỗi SQL."
            elif "sleep" in technique or "time-based" in technique or "benchmark" in technique:
                strategy_family = "time_based_blind_sqli"
                idea_desc = "Khai thác SQL injection mù dựa trên thời gian chờ phản hồi."
            elif "extractvalue" in technique or "updatexml" in technique or "error-based" in technique:
                strategy_family = "error_based_sqli"
                idea_desc = "Khai thác SQL injection dựa trên việc ép cơ sở dữ liệu trả về lỗi có chứa thông tin nhạy cảm."
            elif "union select" in technique or "union-based" in technique:
                strategy_family = "union_based_sqli"
                idea_desc = "Khai thác SQL injection sử dụng toán tử UNION SELECT để lấy dữ liệu."
            elif "boolean" in technique or "or 1=1" in technique:
                strategy_family = "boolean_based_blind_sqli"
                idea_desc = "Khai thác SQL injection mù dựa trên kết quả đúng/sai (true/false)."
            elif "comment" in technique:
                strategy_family = "comment_obfuscation_sqli"
                idea_desc = "Ẩn dấu lệnh SQL bằng cách chèn các ký tự comment SQL."
            elif "stacked" in technique or "xp_cmdshell" in technique:
                strategy_family = "stacked_queries_sqli"
                idea_desc = "Thực thi nhiều lệnh SQL liên tiếp trong một truy vấn."
            elif "path traversal" in technique or "file inclusion" in technique:
                strategy_family = "path_traversal_sqli"
                idea_desc = "Khai thác lỗi path traversal/file inclusion thông qua SQLi."
            elif "character encode" in technique or "charset mix" in technique or "unicode" in technique:
                strategy_family = "character_encoding_sqli"
                idea_desc = "Thay đổi mã hóa ký tự để né tránh bộ lọc WAF."
        
        # XSS Strategies
        elif attack_type == "XSS":
            if "onerror" in technique or "onload" in technique or "event" in technique or "onfocus" in technique or "onmouseover" in technique or "onplay" in technique or "onbefore" in technique or "onactivate" in technique or "ondblclick" in technique:
                strategy_family = "event_handler_xss"
                idea_desc = "Khai thác XSS thông qua các trình xử lý sự kiện HTML/JS."
            elif "script" in technique or "script tag" in technique:
                strategy_family = "script_tag_xss"
                idea_desc = "Khai thác XSS sử dụng thẻ <script>."
            elif "image" in technique or "img" in technique or "svg" in technique:
                strategy_family = "image_svg_xss"
                idea_desc = "Khai thác XSS thông qua các thẻ hình ảnh hoặc SVG."
            elif "iframe" in technique or "frame" in technique or "object" in technique or "embed" in technique:
                strategy_family = "iframe_object_embed_xss"
                idea_desc = "Khai thác XSS sử dụng thẻ iframe, object hoặc embed."
            elif "html entity" in technique or "html-encode" in technique:
                strategy_family = "html_entity_encoding_xss"
                idea_desc = "Ẩn dấu payload XSS bằng cách mã hóa HTML entity."
            elif "url encode" in technique or "double url" in technique or "triple url" in technique:
                strategy_family = "url_encoding_xss"
                idea_desc = "Mã hóa URL payload XSS để né tránh bộ lọc."
            elif "dom-based" in technique:
                strategy_family = "dom_based_xss"
                idea_desc = "Khai thác XSS thông qua các thao tác trên DOM."

        # OS_INJECTION Strategies
        elif attack_type == "OS_INJECTION":
            if "command" in technique or "ls" in technique or "cat" in technique or "whoami" in technique or "id" in technique or "ping" in technique:
                strategy_family = "basic_command_execution_os"
                idea_desc = "Thực thi các lệnh hệ thống cơ bản."
            elif "obfuscation" in technique or "encode" in technique or "whitespace" in technique:
                strategy_family = "obfuscated_os_command"
                idea_desc = "Ẩn dấu lệnh hệ thống bằng các kỹ thuật mã hóa/obfuscation."
        
        # Default / Unclassified
        if not strategy_family:
            strategy_family = "unclassified_general"
            idea_desc = f"Chiến lược {attack_type} tổng quát, chưa phân loại cụ thể."

        strategy_groups[(attack_type, strategy_family)]["patterns"].append(p["doc_id"])
        strategy_groups[(attack_type, strategy_family)]["attack_type"] = attack_type
        strategy_groups[(attack_type, strategy_family)]["idea"] = idea_desc
        
    strategies_docs = []
    for idx, (strategy_key_tuple, data) in enumerate(strategy_groups.items()):
        attack_type, family_name = strategy_key_tuple # Unpack tuple
        
        # Collect example payloads from constituent patterns
        example_payloads = []
        for pattern_doc_id in data["patterns"]:
            # Find the pattern doc to get its example payloads
            original_pattern_doc = next((p for p in patterns if p["doc_id"] == pattern_doc_id), None)
            if original_pattern_doc:
                example_payloads.extend(original_pattern_doc["example_payloads"])
        
        # Ensure unique payloads and limit to 5
        example_payloads = list(set(example_payloads))[:5]

        doc = {
            "doc_id": f"strategy_{attack_type}_{family_name}_{idx}",
            "kind": "attack_strategy",
            "attack_type": attack_type,
            "idea": data["idea"],
            "from_patterns": data["patterns"],
            "notes": f"Chiến lược rút ra từ {len(data['patterns'])} pattern khác nhau về {family_name}.",
            "example_payloads": example_payloads,
            "family_name": family_name # Add family_name for easier access in red_rag_integration
        }
        strategies_docs.append(doc)
        
    os.makedirs(os.path.dirname(output_strategies_file), exist_ok=True)
    with open(output_strategies_file, 'w', encoding='utf-8') as f:
        for doc in strategies_docs:
            f.write(json.dumps(doc) + "\n")
            
    print(f"Generated {len(strategies_docs)} attack strategy documents to {output_strategies_file}")
    return len(strategies_docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern_file", default="data/rag/internal_payload_patterns.jsonl")
    parser.add_argument("--output_strategies_file", default="data/rag/internal_attack_strategies.jsonl")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/red_rag_generate_strategies_from_patterns.py"
    
    try:
        count = generate_strategies(args.input_pattern_file, args.output_strategies_file)
        if count is not None:
            log_message(cmd_str, "OK", args.output_strategies_file)
        else:
            log_message(cmd_str, "FAIL", "Input pattern file not found")
    except Exception as e:
        print(f"Error: {e}")
        log_message(cmd_str, "FAIL", str(e))