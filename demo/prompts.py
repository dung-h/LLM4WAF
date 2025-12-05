# demo/prompts.py

def build_phase1_prompt(attack_type: str, technique: str) -> str:
    """
    Builds a simplified prompt for Phase 1 SFT models.
    """
    return (
        f"Generate a {attack_type} payload using {technique} to bypass WAF. "
        "Output ONLY the payload string. Do NOT add explanations or comments. "
        "Do NOT wrap it in code fences."
    )

def build_phase2_prompt(attack_type: str, technique: str) -> str:
    """
    Builds a structured prompt for Phase 2/3 Reasoning models.
    This template includes context and an explicit technique to guide the model.
    """
    waf_type = "ModSecurity + OWASP CRS 3.3 (PL1)" # Default for demo
    injection_point = "query parameter"
    history_list = "None" # For simplicity in demo, assuming no history
    
    prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {waf_type}.
- Attack type: {attack_type}.
- Injection point: {injection_point}.

Previously tried payloads against this target (all blocked by the WAF):
{history_list}

Your task:
Generate a NEW {attack_type} payload that has a higher chance of bypassing this WAF while still reaching and triggering the underlying vulnerability.
Use the following core technique as the main idea for the payload:
- Technique: {technique}

You may combine this technique with additional obfuscation tricks if it helps evade the filter, but keep the payload compact and realistic.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations or comments.
- Do NOT wrap it in code fences."""
    
    return prompt

def generate_full_prompt_from_template(model_name: str, prompt_mode: str, custom_prompt: str, attack_type: str, technique: str) -> str:
    """
    Generates the final prompt string based on selected mode and model.
    """
    if prompt_mode == "Custom":
        return custom_prompt
    elif prompt_mode == "Auto (Standard - Phase 1)":
        return build_phase1_prompt(attack_type, technique)
    elif prompt_mode == "Auto (Reasoning - Phase 2/3)":
        return build_phase2_prompt(attack_type, technique)
    else:
        # Fallback, should not happen
        return build_phase1_prompt(attack_type, technique)

# --- Common Attack Techniques ---
SQLI_TECHNIQUES = [
    "Double URL Encode", "Comment Obfuscation (/**/)", "Inline Comment Versioning (/*!50000*/)",
    "Hex Encoding", "Whitespace Bypass using Newlines/Tabs", "Boolean-based Blind (AND 1=1)",
    "Time-based Blind (SLEEP/BENCHMARK)", "Union Select with Null Bytes",
    "Case Manipulation (SeLeCt/UnIoN)", "Tautology with Arithmetic (AND 10-2=8)"
]

XSS_TECHNIQUES = [
    "SVG Event Handler", "Unicode Normalization", "IMG Tag with OnError",
    "Body Tag with OnLoad", "Javascript Pseudo-protocol in A Tag",
    "Case Manipulation (<ScRiPt>)", "Attribute Injection (breaking out of quotes)"
]

OS_INJECTION_TECHNIQUES = [
    "Command Concatenation (; || &&)", "Variable Expansion Obfuscation", "Base64 Encoding Wrapper"
]

ALL_TECHNIQUES = {
    "SQL Injection": SQLI_TECHNIQUES,
    "XSS": XSS_TECHNIQUES,
    "OS Injection": OS_INJECTION_TECHNIQUES,
    "Random/Auto": ["Random/Auto"] # Placeholder for random selection
}
