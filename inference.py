"""
inference.py — Stable Agent (No Tool Calling, JSON/Text Parsing)
"""

import os
import sys
import json
import random
import re
from dotenv import load_dotenv
from openai import OpenAI

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
TEMPERATURE = 0.2
MAX_STEPS = 15
MAX_HISTORY = 10

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a robot task agent.

Your job is to choose the NEXT best action.

IMPORTANT:
- ONLY choose from the given Valid Actions.
- DO NOT invent actions.
- DO NOT explain.

OUTPUT FORMAT (STRICT JSON ONLY):
{"action_name": "<one valid action>"}
"""

def extract_action(response_text, valid_actions):
    """Try JSON first, then fallback to text matching"""

    # Try JSON parsing
    try:
        data = json.loads(response_text)
        action = data.get("action_name")
        if action in valid_actions:
            return action
    except:
        pass

    # Fallback: regex or substring match
    for action in valid_actions:
        if action in response_text:
            return action

    return None


def run_episode(task_id: str):
    env = CosmicBytesEnvironment(task_id=task_id)
    obs = env.reset()

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    steps = 0
    rewards = []
    success = False

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while steps < MAX_STEPS and not getattr(obs, "done", False):

        actions_text = ", ".join(obs.available_actions) if obs.available_actions else "None"

        user_msg = (
            f"Task: {obs.task_description}\n"
            f"Valid Actions: {actions_text}\n"
            f"Return JSON only."
        )

        content = [{"type": "text", "text": user_msg}]

        # Add image only once
        if getattr(obs, "image_base64", None) and steps == 0:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{obs.image_base64}"
                }
            })

        messages.append({"role": "user", "content": content})

        chosen_action = None

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
            )

            resp_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": resp_text})

            chosen_action = extract_action(resp_text, obs.available_actions)

        except Exception as e:
            print(f"[ERROR] LLM call failed: {str(e)}")
            continue  # retry same step

        # Fallback if extraction fails
        if not chosen_action:
            if obs.available_actions:
                chosen_action = random.choice(obs.available_actions)
            else:
                chosen_action = "noop"

        # Step increment ONLY on valid execution
        steps += 1

        # Execute action
        action = CosmicBytesAction(step_sequence=[chosen_action])
        obs = env.step(action)

        rewards.append(obs.reward)

        # Feedback into memory
        messages.append({
            "role": "user",
            "content": f"Result: {obs.task_description}"
        })

        # Trim memory
        if len(messages) > MAX_HISTORY:
            messages = [messages[0]] + messages[-(MAX_HISTORY - 1):]

        print(f"[STEP] step={steps} action={chosen_action} reward={obs.reward:.2f} done={obs.done}")

        if getattr(obs, "done", False):
            success = "SUCCESS" in obs.task_description
            break

    print(
        f"[END] success={success} steps={steps} "
        f"rewards={','.join([f'{r:.2f}' for r in rewards])}"
    )


def main():
    if not API_KEY:
        print("ERROR: API_KEY must be set.")
        sys.exit(1)

    for task_id in TASKS.keys():
        run_episode(task_id)


if __name__ == "__main__":
    main()