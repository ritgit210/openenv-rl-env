"""
inference.py — Robust Stable Agent (Async, Timeouts, Rate-Limit Support)
"""

import os
import sys
import json
import random
import asyncio
import math
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Allow imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmic_bytes.server.cosmic_bytes_environment import CosmicBytesEnvironment, TASKS
from cosmic_bytes.models import CosmicBytesAction

# CONFIG
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "cosmic-bytes"
TEMPERATURE = 0.1
MAX_STEPS = 12
MAX_HISTORY = 8
TIMEOUT = 25.0

client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=TIMEOUT)

SYSTEM_PROMPT = """You are a robot task agent. Follow instructions exactly and return JSON."""

# Validator requires strict open interval (0, 1) per task.
_SCORE_EPSILON = 1e-3


def _normalize_task_score(value: float) -> float:
    """Return a finite score guaranteed to be strictly inside (0, 1)."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return _SCORE_EPSILON
    return float(max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, value)))

def extract_action(response_text, valid_actions):
    try:
        clean_text = response_text.strip()
        if "```" in clean_text:
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"): clean_text = clean_text[4:]
        data = json.loads(clean_text)
        action = data.get("action_name")
        if action in valid_actions: return action
    except:
        pass
    for action in valid_actions:
        if action in response_text: return action
    return None

async def run_episode(task_id: str):
    env = CosmicBytesEnvironment(task_id=task_id)
    obs = env.reset()

    print(f"[START] task={task_id}")
    steps = 0
    rewards = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while steps < MAX_STEPS and not getattr(obs, "done", False):
        user_msg = f"Goal: {obs.task_description}\nValid: {', '.join(obs.available_actions)}\nAction JSON:"
        content = [{"type": "text", "text": user_msg}]
        if steps == 0 and getattr(obs, "image_base64", None):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs.image_base64}"}})
        
        messages.append({"role": "user", "content": content})
        
        chosen_action = None
        retries = 0
        retry_delay = 1.0
        
        while retries < 5:  # High retries for rate limits
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, temperature=TEMPERATURE
                )
                resp_text = response.choices[0].message.content
                messages.append({"role": "assistant", "content": resp_text})
                chosen_action = extract_action(resp_text, obs.available_actions)
                break
            except Exception as e:
                err_msg = str(e).lower()
                retries += 1
                if "rate_limit" in err_msg or "429" in err_msg:
                    delay = retry_delay * (2 ** (retries - 1)) + random.uniform(0, 1)
                    print(f"[RETRY {retries}/5] rate_limited: task={task_id} step={steps+1} sleep={delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"[RETRY {retries}/5] error: task={task_id} step={steps+1} error={str(e)[:50]}")
                    await asyncio.sleep(1)

        if not chosen_action:
            chosen_action = random.choice(obs.available_actions) if obs.available_actions else "noop"

        steps += 1
        obs = env.step(CosmicBytesAction(step_sequence=[chosen_action]))
        rewards.append(obs.reward)
        messages.append({"role": "user", "content": f"Result: {obs.task_description}"})
        if len(messages) > MAX_HISTORY:
            messages = [messages[0]] + messages[-(MAX_HISTORY - 1):]
        
        print(f"[STEP] task={task_id} step={steps} action={chosen_action} reward={obs.reward:.2f} done={obs.done}")

    success = "SUCCESS" in obs.task_description
    # Sum step rewards, then enforce strict validator-safe open interval.
    task_score = _normalize_task_score(sum(rewards))
    print(f"[END] task={task_id} success={success} steps={steps} task_score={task_score:.4f}")
    return task_score


async def main():
    if not API_KEY:
        print("API_KEY missing.")
        sys.exit(1)

    scores = {}
    # Sequential execution – avoids rate-limit spikes across tasks
    for task_id in TASKS.keys():
        try:
            scores[task_id] = _normalize_task_score(await run_episode(task_id))
        except Exception as e:
            print(f"[WARN] task={task_id} failed during rollout: {str(e)[:120]}")
            # Keep submission valid even if one task rollout fails.
            scores[task_id] = _SCORE_EPSILON

    # Final compliance gate: never emit boundary values to validator.
    for task_id, raw_score in list(scores.items()):
        normalized = _normalize_task_score(raw_score)
        if normalized != raw_score:
            print(f"[WARN] normalized out-of-range score for {task_id}: {raw_score} -> {normalized}")
        scores[task_id] = normalized

    print(f"\n[SCORES] {scores}")
    return scores


if __name__ == "__main__":
    asyncio.run(main())
