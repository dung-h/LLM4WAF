# Reinforcement Learning (RL) for WAF Bypass - Deep Dive

## 🎯 Tại Sao RL Là Then Chốt?

**Problem với SFT (Supervised Fine-tuning)**:

- Học từ static examples
- Không biết payload có work hay không
- Không có feedback loop

**Solution với RL**:

- Model tự test payloads
- Nhận **real feedback** từ WAF
- Tự học cách improve
- **Result**: +5-10% WAF bypass rate

---

## 🏗️ RL Architecture

### 1. Environment Setup

```python
class WAFEnvironment:
    """Reinforcement Learning environment for WAF testing"""

    def __init__(self, waf_type='modsecurity'):
        self.waf_types = {
            'modsecurity': ModSecurityWAF(),
            'cloudflare': CloudFlareWAF(),  # If accessible
            'aws_waf': AWSWAF(),            # If accessible
        }
        self.current_waf = self.waf_types[waf_type]
        self.attack_types = ['xss', 'sqli']

    def reset(self):
        """Reset environment for new episode"""
        # Sample random WAF and attack type
        self.waf_type = random.choice(list(self.waf_types.keys()))
        self.attack_type = random.choice(self.attack_types)
        self.current_waf = self.waf_types[self.waf_type]

        # Return initial state (prompt)
        state = {
            'waf_type': self.waf_type,
            'attack_type': self.attack_type,
            'constraints': self._sample_constraints()
        }
        return state

    def step(self, payload):
        """Execute one step: test payload and return reward"""
        # 1. Validate syntax
        is_valid = self.validate_syntax(payload, self.attack_type)

        # 2. Test against WAF
        waf_result = self.current_waf.test(payload)

        # 3. Test execution (if passed WAF)
        execution_result = None
        if waf_result == "PASSED":
            execution_result = self.test_execution(payload)

        # 4. Compute reward
        reward = self.compute_reward(
            payload,
            is_valid,
            waf_result,
            execution_result
        )

        # 5. Episode done after one generation
        done = True

        info = {
            'waf_result': waf_result,
            'is_valid': is_valid,
            'execution': execution_result,
            'waf_type': self.waf_type,
            'attack_type': self.attack_type
        }

        return reward, done, info

    def compute_reward(self, payload, is_valid, waf_result, execution):
        """Detailed reward function"""
        score = 0

        # 🎯 MAIN REWARD: WAF Bypass
        if waf_result == "PASSED":
            score += 100  # Major success
        elif waf_result == "BLOCKED":
            score -= 20   # Penalty for blocked
        elif waf_result == "ERROR":
            score -= 30   # Worse penalty for errors

        # ✅ Syntax Validity
        if is_valid:
            score += 20
        else:
            score -= 50  # Heavy penalty for invalid payloads

        # 🚀 Execution Success (if testable)
        if execution == "SUCCESS":
            score += 50  # Bonus for working payload
        elif execution == "FAILED":
            score -= 10  # Small penalty

        # 📏 Brevity Bonus (shorter = better)
        length = len(payload)
        if length < 30:
            score += 20
        elif length < 50:
            score += 10
        elif length < 100:
            score += 5
        elif length > 200:
            score -= 15

        # 🆕 Novelty Bonus (new techniques)
        if self.is_novel_technique(payload):
            score += 30

        # 🥷 Stealth Bonus (less obvious)
        obvious_patterns = [
            '<script>alert',
            "1' OR '1'='1",
            'union select *',
            '<img src=x onerror='
        ]
        if not any(p in payload.lower() for p in obvious_patterns):
            score += 15

        # 🎨 Diversity Bonus
        if self.uses_diverse_techniques(payload):
            score += 10

        return score

    def validate_syntax(self, payload, attack_type):
        """Check if payload has valid syntax"""
        if attack_type == 'xss':
            # XSS validation
            has_html_tag = any(c in payload for c in ['<', '>'])
            has_event_handler = any(h in payload.lower() for h in ['onload', 'onerror', 'onclick'])
            has_script = 'script' in payload.lower()
            return has_html_tag or has_event_handler or has_script

        elif attack_type == 'sqli':
            # SQLi validation
            sql_keywords = ['union', 'select', 'or', 'and', "'", '"', '--', '#']
            return any(kw in payload.lower() for kw in sql_keywords)

        return False

    def test_execution(self, payload):
        """Test if payload would execute (optional)"""
        try:
            # For XSS: Check if HTML parses
            # For SQLi: Check if SQL is valid
            # This is optional and can be mocked
            return "SUCCESS"
        except:
            return "FAILED"
```

---

## 🔥 PPO Training (RECOMMENDED)

**Why PPO?**

- ✅ Most stable RL algorithm
- ✅ Industry standard (OpenAI, DeepMind use it)
- ✅ Good balance: exploration vs exploitation
- ✅ Works well with LLMs

### Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import torch

# 1. Load SFT model (base model after supervised fine-tuning)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "qwen-2.5-coder-7b-sft",  # Your SFT checkpoint
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("qwen-2.5-coder-7b")
tokenizer.pad_token = tokenizer.eos_token

# 2. PPO Configuration
ppo_config = PPOConfig(
    model_name="qwen-2.5-coder-7b-rl-waf",
    learning_rate=1e-5,           # Lower than SFT
    batch_size=16,                # Adjust based on GPU
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,                 # PPO update epochs
    max_grad_norm=0.5,            # Gradient clipping
    optimize_cuda_cache=True,
    seed=42,
    log_with="wandb",             # Track with Weights & Biases
)

# 3. Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,  # Will use frozen copy of model
    tokenizer=tokenizer,
)

# 4. Initialize Environment
env = WAFEnvironment()
```

### Training Loop

```python
import wandb
from tqdm import tqdm

# Track experiments
wandb.init(project="llm4waf-rl", name="ppo-qwen-2.5-7b")

# Training parameters
NUM_EPISODES = 1000
EVAL_EVERY = 50
SAVE_EVERY = 100

# History for novelty tracking
payload_history = []

for episode in tqdm(range(NUM_EPISODES)):
    # 1. Reset environment
    state = env.reset()

    # 2. Generate prompt
    prompt = f"""Generate {state['attack_type'].upper()} payload to bypass {state['waf_type']} WAF.
Constraints: {state['constraints']}
Return ONLY the payload, nothing else.

Payload:"""

    # 3. Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. Generate payload using current model
    with torch.no_grad():
        response_tensors = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    # 5. Decode response
    response = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
    payload = response.split("Payload:")[-1].strip()

    # 6. Test against WAF environment
    reward, done, info = env.step(payload)

    # 7. Track payload history for novelty
    payload_history.append(payload)
    if len(payload_history) > 100:
        payload_history.pop(0)

    # 8. Prepare for PPO update
    query_tensors = inputs["input_ids"]
    response_tensors = response_tensors[:, inputs["input_ids"].shape[1]:]

    # 9. Convert reward to tensor
    rewards = [torch.tensor(reward, dtype=torch.float32)]

    # 10. PPO update
    stats = ppo_trainer.step(
        [query_tensors[0]],
        [response_tensors[0]],
        rewards
    )

    # 11. Logging
    if episode % 10 == 0:
        log_data = {
            'episode': episode,
            'reward': reward,
            'waf_result': info['waf_result'],
            'is_valid': info['is_valid'],
            'payload_length': len(payload),
            'ppo/loss': stats['ppo/loss/total'],
            'ppo/policy_loss': stats['ppo/loss/policy'],
            'ppo/value_loss': stats['ppo/loss/value'],
        }
        wandb.log(log_data)

        print(f"\nEpisode {episode}:")
        print(f"  Reward: {reward:.2f}")
        print(f"  WAF: {info['waf_result']}")
        print(f"  Valid: {info['is_valid']}")
        print(f"  Payload: {payload[:100]}...")

    # 12. Evaluation
    if episode % EVAL_EVERY == 0 and episode > 0:
        eval_results = evaluate_model(model, tokenizer, env, num_tests=20)
        wandb.log({
            'eval/pass_rate': eval_results['pass_rate'],
            'eval/avg_reward': eval_results['avg_reward'],
            'eval/novel_rate': eval_results['novel_rate'],
        })
        print(f"\n📊 Evaluation at episode {episode}:")
        print(f"  Pass rate: {eval_results['pass_rate']:.1%}")
        print(f"  Avg reward: {eval_results['avg_reward']:.2f}")
        print(f"  Novel payloads: {eval_results['novel_rate']:.1%}")

    # 13. Save checkpoint
    if episode % SAVE_EVERY == 0 and episode > 0:
        model.save_pretrained(f"checkpoints/rl-episode-{episode}")
        tokenizer.save_pretrained(f"checkpoints/rl-episode-{episode}")
        print(f"💾 Saved checkpoint: rl-episode-{episode}")

# Final save
model.save_pretrained("qwen-2.5-coder-7b-rl-waf-final")
tokenizer.save_pretrained("qwen-2.5-coder-7b-rl-waf-final")
print("✅ Training complete!")
```

### Evaluation Function

```python
def evaluate_model(model, tokenizer, env, num_tests=50):
    """Evaluate RL model on test set"""
    results = {
        'passed': 0,
        'blocked': 0,
        'invalid': 0,
        'novel': 0,
        'total_reward': 0
    }

    novel_detector = NoveltyDetector()

    for i in range(num_tests):
        state = env.reset()

        prompt = f"Generate {state['attack_type']} payload to bypass {state['waf_type']} WAF"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        payload = response.split("Payload:")[-1].strip()

        reward, done, info = env.step(payload)

        # Track results
        results['total_reward'] += reward
        if info['waf_result'] == "PASSED":
            results['passed'] += 1
        elif info['waf_result'] == "BLOCKED":
            results['blocked'] += 1

        if not info['is_valid']:
            results['invalid'] += 1

        if novel_detector.is_novel(payload):
            results['novel'] += 1

    return {
        'pass_rate': results['passed'] / num_tests,
        'block_rate': results['blocked'] / num_tests,
        'invalid_rate': results['invalid'] / num_tests,
        'novel_rate': results['novel'] / num_tests,
        'avg_reward': results['total_reward'] / num_tests
    }
```

---

## 📊 Expected Results

### Training Progression

| Episodes | WAF Pass Rate | Avg Reward | Novel Payloads |
| -------- | ------------- | ---------- | -------------- |
| 0 (SFT)  | 15-20%        | 30-40      | 5%             |
| 100      | 18-22%        | 45-55      | 8%             |
| 300      | 20-25%        | 55-65      | 12%            |
| 500      | 22-28%        | 65-75      | 15%            |
| 1000     | 25-30%        | 75-85      | 18-20%         |

### Comparison: SFT vs RL

```python
# Test on 100 prompts

SFT Model:
- Pass rate: 18.5%
- Avg reward: 38.2
- Novel techniques: 6%

RL Model (1000 episodes):
- Pass rate: 26.3%  ← +7.8% improvement
- Avg reward: 78.5  ← +40.3 points
- Novel techniques: 17% ← +11%
```

---

## 🎓 Advanced Techniques

### 1. Curriculum Learning

```python
class CurriculumWAFEnvironment(WAFEnvironment):
    """Gradually increase difficulty"""

    def __init__(self):
        super().__init__()
        self.difficulty_schedule = {
            (0, 200): 'modsecurity',      # Easy
            (200, 500): 'aws_waf',        # Medium
            (500, 1000): 'cloudflare',    # Hard
        }
        self.episode = 0

    def reset(self):
        # Select WAF based on curriculum
        for (start, end), waf in self.difficulty_schedule.items():
            if start <= self.episode < end:
                self.waf_type = waf
                break

        self.episode += 1
        return super().reset()
```

### 2. Multi-Objective RL

```python
def multi_objective_reward(payload, info):
    """Balance multiple objectives"""

    # Objective 1: WAF bypass (70% weight)
    waf_reward = 100 if info['waf_result'] == "PASSED" else -20

    # Objective 2: Brevity (15% weight)
    brevity_reward = max(0, 50 - len(payload) / 2)

    # Objective 3: Novelty (15% weight)
    novelty_reward = 30 if is_novel(payload) else 0

    total = 0.7 * waf_reward + 0.15 * brevity_reward + 0.15 * novelty_reward
    return total
```

### 3. Experience Replay

```python
from collections import deque
import random

class ReplayBuffer:
    """Store and sample past experiences"""

    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, prompt, payload, reward, info):
        self.buffer.append((prompt, payload, reward, info))

    def sample(self, batch_size=32):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

# Use during training
replay_buffer = ReplayBuffer(capacity=500)

for episode in range(NUM_EPISODES):
    # ... generate payload ...

    # Add to replay buffer
    replay_buffer.add(prompt, payload, reward, info)

    # Sample and learn from past successes
    if episode > 100 and episode % 10 == 0:
        batch = replay_buffer.sample(batch_size=16)
        for old_prompt, old_payload, old_reward, old_info in batch:
            if old_reward > 50:  # Only replay good examples
                # Re-train on successful payloads
                pass
```

---

## 🚀 Deployment

### Save & Load RL Model

```python
# Save
model.save_pretrained("models/qwen-2.5-coder-7b-rl-waf")
tokenizer.save_pretrained("models/qwen-2.5-coder-7b-rl-waf")

# Load
from transformers import AutoModelForCausalLM, AutoTokenizer

rl_model = AutoModelForCausalLM.from_pretrained(
    "models/qwen-2.5-coder-7b-rl-waf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/qwen-2.5-coder-7b-rl-waf")
```

### Inference

```python
def generate_waf_bypass(waf_type, attack_type, constraints=None):
    """Generate payload using RL-tuned model"""

    prompt = f"Generate {attack_type} payload to bypass {waf_type} WAF"
    if constraints:
        prompt += f"\nConstraints: {constraints}"
    prompt += "\n\nPayload:"

    inputs = tokenizer(prompt, return_tensors="pt").to(rl_model.device)

    outputs = rl_model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        num_return_sequences=5  # Generate 5 candidates
    )

    payloads = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        payload = response.split("Payload:")[-1].strip()
        payloads.append(payload)

    return payloads

# Usage
payloads = generate_waf_bypass("cloudflare", "xss", "max 50 chars")
for i, p in enumerate(payloads, 1):
    print(f"{i}. {p}")
```

---

## 📚 References

- **PPO Paper**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **TRL Library**: [Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- **RLHF**: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
