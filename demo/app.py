# demo/app.py
import gradio as gr
import pandas as pd
import os
import time
import logging
import sys
from typing import List, Tuple, Dict, Any

# Adjust path to import modules from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.model_loader import model_loader
from demo.waf_executor import waf_executor
from demo.prompts import SQLI_TECHNIQUES, XSS_TECHNIQUES, OS_INJECTION_TECHNIQUES, ALL_TECHNIQUES, generate_full_prompt_from_template

# --- Configuration ---
# Models available for selection
AVAILABLE_MODELS = {
    "Phi-3 Mini": {
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "phases": {
            "Phase 1 SFT": "experiments/remote_phi3_mini_phase1",
            "Phase 3 RL": "experiments/remote_phi3_mini_phase3_rl"
        }
    },
    "Qwen 3B": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "phases": {
            "Phase 1 SFT": "experiments/remote_qwen_3b_phase1",
            "Phase 3 RL": "experiments/remote_qwen_3b_phase3_rl"
        }
    },
    "Gemma 2 2B": {
        "base_model": "google/gemma-2-2b-it",
        "phases": {
            "Phase 1 SFT": "experiments/remote_gemma2_2b_phase1",
            "Phase 3 RL": "experiments/remote_gemma2_2b_phase3_rl"
        }
    }
}

# --- Logging setup for Gradio ---
# This ensures logs appear in the Gradio console output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Gradio UI State ---
current_attack_results = []

# --- Functions for Gradio UI ---

def get_adapter_options(base_model_name: str) -> gr.Dropdown:
    """Updates the adapter dropdown based on the selected base model."""
    if base_model_name in AVAILABLE_MODELS:
        phases = list(AVAILABLE_MODELS[base_model_name]["phases"].keys())
        return gr.Dropdown(choices=phases, value=phases[0] if phases else None)
    return gr.Dropdown(choices=[], value=None)

def load_model_for_demo(base_model_name: str, phase_name: str) -> Tuple[str, str]:
    """Loads the selected LLM model and adapter."""
    if not base_model_name or not phase_name:
        return "ERROR: Please select a Base Model and a Phase.", "unloaded"
    
    model_config = AVAILABLE_MODELS.get(base_model_name)
    if not model_config:
        return f"ERROR: Model config not found for {base_model_name}.", "unloaded"
    
    adapter_path = model_config["phases"].get(phase_name)
    if not adapter_path or not os.path.exists(adapter_path):
        return f"ERROR: Adapter path not found or invalid: {adapter_path}", "unloaded"
    
    try:
        model, tokenizer = model_loader.load_model(model_config["base_model"], adapter_path)
        if model and tokenizer:
            return f"Model '{base_model_name} - {phase_name}' loaded successfully!", "loaded"
        else:
            return "ERROR: Failed to load model for unknown reason.", "unloaded"
    except Exception as e:
        return f"ERROR: Failed to load model: {e}", "unloaded"

def unload_current_model() -> Tuple[str, str]:
    """Unloads the currently active LLM model."""
    model_loader.unload_model()
    return "Model unloaded.", "unloaded"

def get_attack_technique_options(attack_type: str) -> gr.Dropdown:
    """Updates the technique dropdown based on the selected attack type."""
    return gr.Dropdown(choices=ALL_TECHNIQUES.get(attack_type, []), value=None)

def update_prompt_ui(prompt_mode: str) -> gr.Textbox:
    """Shows/hides custom prompt textbox."""
    if prompt_mode == "Custom":
        return gr.Textbox(visible=True)
    return gr.Textbox(visible=False)

def verify_target_waf(target_mode: str, remote_base_url: str, remote_username: str, remote_password: str, remote_target_param: str) -> str:
    """Verifies connection to the specified WAF target."""
    # Normalize user input
    base_url = (remote_base_url or "").strip().rstrip("/")
    username = (remote_username or "").strip()
    password = (remote_password or "").strip()
    target_param = (remote_target_param or "id").strip() or "id"

    if target_mode == "Local (DVWA Docker)":
        # Allow overriding base_url/creds; fallback to defaults
        base_url = base_url or "http://localhost:8000"
        username = username or "admin"
        password = password or "password"
        config = {
            "base_url": base_url,
            "username": username,
            "password": password,
            "target_param": target_param,
            "login_required": True
        }
    else: # Remote (Custom URL)
        if not base_url:
            return "ERROR: Remote URL cannot be empty."
        config = {
            "base_url": base_url,
            "username": username,
            "password": password,
            "target_param": target_param,
            "login_required": bool(username and password)
        }
    
    waf_executor.update_config(config)
    status_message = waf_executor.verify_target()
    return status_message

def generate_and_attack(
    base_model_name: str, phase_name: str,
    attack_type: str, technique: str,
    prompt_mode: str, custom_prompt: str,
    temperature: float, max_new_tokens: int, loop_count: int,
    progress=gr.Progress()
) -> Tuple[List[List[Any]], str]:
    """Generates payloads and attacks the WAF."""
    global current_attack_results
    current_attack_results = [] # Reset results for new run
    
    model, tokenizer, loaded_model_id, loaded_adapter_path = model_loader.get_loaded_model()
    
    if not model or loaded_model_id != AVAILABLE_MODELS[base_model_name]["base_model"] or loaded_adapter_path != AVAILABLE_MODELS[base_model_name]["phases"][phase_name]:
        return current_attack_results, "ERROR: Model not loaded or mismatched. Please load the correct model first."

    if not attack_type:
        return current_attack_results, "ERROR: Attack Type cannot be empty."

    log_messages = []
    
    for i in progress.tqdm(range(loop_count), desc="Generating & Attacking"):
        log_messages.append(f"--- Running Loop {i+1}/{loop_count} ---")
        
        # 1. Prepare Prompt
        current_technique = technique
        if technique == "Random/Auto":
            current_technique = random.choice(ALL_TECHNIQUES.get(attack_type, [""])) # Select a random technique if Auto
            if not current_technique:
                log_messages.append(f"ERROR: No techniques found for {attack_type} or Random/Auto failed.")
                current_attack_results.append([i+1, "", "ERROR", "-", "-", "No technique"])
                yield current_attack_results, "\n".join(log_messages)
                continue

        prompt_text = generate_full_prompt_from_template(base_model_name, prompt_mode, custom_prompt, attack_type, current_technique)
        log_messages.append(f"Prompting with technique: {current_technique}")

        # 2. Generate Payload
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False # For gradient checkpointing compatibility, safe for inference
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Robust cleaning logic (from scripts/evaluate_remote_adapters.py)
            payload_raw = generated_text.strip()
            markers = ["<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>", "<|assistant|>", "### Response:", "Assistant:"]
            for marker in markers:
                if marker in payload_raw:
                    payload_raw = payload_raw.split(marker)[-1].strip()
                    break 
            artifacts = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "<|end|>", "</s>", tokenizer.eos_token if tokenizer.eos_token else ""]
            for artifact in artifacts:
                if artifact:
                    payload_raw = payload_raw.replace(artifact, "")
            payload_raw = payload_raw.strip()
            if "```" in payload_raw:
                parts = payload_raw.split("```")
                if len(parts) > 1:
                    payload_raw = parts[1].strip()
                    for lang in ["sql", "html", "bash", "javascript", "xml"]:
                        if payload_raw.lower().startswith(lang + "\n"):
                            payload_raw = payload_raw[len(lang) + 1:].strip()
                        elif payload_raw.lower().startswith(lang + " "): 
                            payload_raw = payload_raw[len(lang) + 1:].strip()

            log_messages.append(f"Generated payload: {payload_raw}")

        except Exception as e:
            log_messages.append(f"ERROR during payload generation: {e}")
            current_attack_results.append([i+1, "ERROR", "ERROR", "-", "-", f"Generation Failed: {e}"])
            yield current_attack_results, "\n".join(log_messages)
            continue

        # 3. Attack WAF
        try:
            attack_result = waf_executor.execute_attack(payload_raw, attack_type)
            current_attack_results.append([
                i+1,
                payload_raw,
                attack_result["status"],
                attack_result["http_code"],
                f"{attack_result['latency']:.2f} ms",
                attack_result["message"]
            ])
            log_messages.append(f"WAF Result: {attack_result['status']} (HTTP {attack_result['http_code']}) - {attack_result['message']}")
            yield current_attack_results, "\n".join(log_messages) # Update UI in real-time
        except Exception as e:
            log_messages.append(f"ERROR during WAF attack: {e}")
            current_attack_results.append([i+1, payload_raw, "ERROR", "-", "-", f"WAF Attack Failed: {e}"])
            yield current_attack_results, "\n".join(log_messages)

    log_messages.append("--- Attack Session Complete ---")
    return current_attack_results, "\n".join(log_messages)

# --- Quick Self-Test Helper (non-UI) ---
def quick_self_test(
    target_url: str = "http://localhost:8000",
    username: str = "admin",
    password: str = "password",
) -> Dict[str, Any]:
    """
    Lightweight sanity check (no model load):
      - Verify adapter paths exist for all models/phases.
      - Ping target_url via WAF executor verify flow.
    Usage (manual):
      >>> from demo.app import quick_self_test
      >>> quick_self_test("http://localhost:8000", "admin", "password")
    """
    adapter_status = []
    for model_name, cfg in AVAILABLE_MODELS.items():
        for phase, path in cfg["phases"].items():
            adapter_status.append({
                "model": model_name,
                "phase": phase,
                "path": path,
                "exists": os.path.exists(path)
            })
    # Verify target
    waf_executor.update_config({
        "base_url": target_url,
        "username": username,
        "password": password,
        "target_param": "id",
        "login_required": True
    })
    target_status = waf_executor.verify_target()
    return {"adapters": adapter_status, "target_status": target_status}

# --- Gradio UI Definition ---
with gr.Blocks(title="LLM4WAF Red Teaming Dashboard") as demo:
    gr.Markdown("# LLM4WAF Red Teaming Dashboard")
    gr.Markdown("Explore LLM-generated WAF evasion payloads.")

    with gr.Row():
        with gr.Column():
            with gr.Group():
                gr.Markdown("### 1. Target WAF Configuration")
                target_mode = gr.Radio(
                    choices=["Local (DVWA Docker)", "Remote (Custom URL)"],
                    value="Local (DVWA Docker)",
                    label="Target Mode"
                )
                
                # Local DVWA Config (Editable defaults)
                with gr.Column(visible=True) as local_dvwa_config:
                    local_base_url = gr.Textbox(value="http://localhost:8000", label="DVWA Base URL", interactive=True)
                    local_username = gr.Textbox(value="admin", label="Username", interactive=True)
                    local_password = gr.Textbox(value="password", label="Password", interactive=True)
                    local_verify_btn = gr.Button("Verify Local DVWA WAF")
                    local_waf_status = gr.Textbox(label="Local WAF Status", interactive=False)

                # Remote Custom Config (Conditional visibility)
                with gr.Column(visible=False) as remote_custom_config:
                    remote_base_url = gr.Textbox(label="Remote Base URL", placeholder="e.g., http://example.com/dvwa")
                    remote_username = gr.Textbox(label="Remote Username (Optional)", placeholder="e.g., admin")
                    remote_password = gr.Textbox(label="Remote Password (Optional)", type="password")
                    remote_verify_btn = gr.Button("Verify Remote Target")
                    remote_waf_status = gr.Textbox(label="Remote WAF Status", interactive=False)
                
                target_mode.change(
                    lambda mode: (
                        gr.update(visible=mode == "Local (DVWA Docker)"),
                        gr.update(visible=mode == "Remote (Custom URL)")
                    ),
                    inputs=[target_mode],
                    outputs=[local_dvwa_config, remote_custom_config]
                )
                
                local_verify_btn.click(
                    verify_target_waf, 
                    inputs=[gr.State("Local (DVWA Docker)"), local_base_url, local_username, local_password, gr.State("id")],
                    outputs=[local_waf_status]
                )
                remote_verify_btn.click(
                    verify_target_waf, 
                    inputs=[target_mode, remote_base_url, remote_username, remote_password, gr.State("id")],
                    outputs=[remote_waf_status]
                )

            with gr.Group():
                gr.Markdown("### 2. Model Selection")
                base_model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()), 
                    label="Base Model", 
                    value="Phi-3 Mini"
                )
                adapter_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS["Phi-3 Mini"]["phases"].keys()),
                    label="Training Phase (Adapter)",
                    value="Phase 3 RL"
                )
                base_model_dropdown.change(
                    get_adapter_options, 
                    inputs=[base_model_dropdown], 
                    outputs=[adapter_dropdown]
                )
                load_model_btn = gr.Button("Load Model")
                unload_model_btn = gr.Button("Unload Model")
                model_status_text = gr.Textbox(label="Model Status", interactive=False, value="Model unloaded.")
                model_status_indicator = gr.State("unloaded") # Internal state for model loaded status

                load_model_btn.click(
                    load_model_for_demo, 
                    inputs=[base_model_dropdown, adapter_dropdown], 
                    outputs=[model_status_text, model_status_indicator]
                )
                unload_model_btn.click(
                    unload_current_model,
                    outputs=[model_status_text, model_status_indicator]
                )

            with gr.Group():
                gr.Markdown("### 3. Attack Configuration")
                attack_type_dropdown = gr.Dropdown(
                    choices=list(ALL_TECHNIQUES.keys()), 
                    label="Attack Type", 
                    value="SQL Injection"
                )
                technique_dropdown = gr.Dropdown(
                    choices=SQLI_TECHNIQUES, 
                    label="Target Technique", 
                    value="Double URL Encode"
                )
                attack_type_dropdown.change(
                    get_attack_technique_options, 
                    inputs=[attack_type_dropdown], 
                    outputs=[technique_dropdown]
                )
            
            with gr.Group():
                gr.Markdown("### 4. Prompt Strategy")
                prompt_mode_radio = gr.Radio(
                    choices=["Auto (Standard - Phase 1)", "Auto (Reasoning - Phase 3)", "Custom"],
                    value="Auto (Reasoning - Phase 3)",
                    label="Prompt Mode"
                )
                custom_prompt_textbox = gr.Textbox(
                    label="Custom Prompt (LLM's instruction)", 
                    placeholder="Enter your custom prompt here...", 
                    lines=5, 
                    visible=False
                )
                prompt_mode_radio.change(
                    update_prompt_ui,
                    inputs=[prompt_mode_radio],
                    outputs=[custom_prompt_textbox]
                )

            with gr.Accordion("Advanced Settings", open=False):
                temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature (Creativity)")
                max_new_tokens_slider = gr.Slider(minimum=64, maximum=512, step=32, value=128, label="Max New Tokens")
                loop_count_slider = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Loop Count (Payloads per run)")
            
            attack_btn = gr.Button("GENERATE & ATTACK", variant="primary")
            gr.Markdown("---")
            gr.Markdown("Made with ❤️ for LLM4WAF Project")

        with gr.Column():
            gr.Markdown("### Live Logs")
            live_logs_output = gr.Textbox(label="Logs", interactive=False, lines=15)
            
            gr.Markdown("### Attack Results")
            attack_results_df = gr.DataFrame(
                headers=["ID", "Payload", "Status", "HTTP Code", "Latency", "Message"],
                row_count=0,
                col_count=6,
                wrap=True,
                interactive=False
            )
            
            with gr.Accordion("Raw Model Output (for debugging)", open=False):
                raw_output_textbox = gr.Textbox(label="Raw Output", interactive=False, lines=10) # Placeholder

    # --- Event Handlers ---
    attack_btn.click(
        generate_and_attack,
        inputs=[
            base_model_dropdown, adapter_dropdown, attack_type_dropdown, technique_dropdown,
            prompt_mode_radio, custom_prompt_textbox,
            temperature_slider, max_new_tokens_slider, loop_count_slider
        ],
        outputs=[attack_results_df, live_logs_output]
    )

demo.launch()
