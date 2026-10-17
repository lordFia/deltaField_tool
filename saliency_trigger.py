import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

examples = [
    "I'm feeling cold.",
    "I'm starting to feel chilly.",
    "Could you make the room warmer?",
    "I'm freezing; can we turn up the heat?"
]

threshold = 0.55
enabled = True

def build_prototype():
    vectors = encoder.encode(examples, normalize_embeddings=True)
    return normalize(vectors.mean(axis=0))

def show_settings():
    print(f"\nTrigger: {'ON' if enabled else 'OFF'}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Concept examples: {len(examples)}")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")

print("""
Interactive Semantic Trigger
Commands:
  :threshold 0.65  Change threshold
  :on              Enable trigger
  :off             Disable trigger
  :add <text>      Add a concept example
  :remove <n>       Remove example by number
  :examples         Show examples
  :settings         Show current settings
  :quit             Exit
""")

prototype = build_prototype()

while True:
    text = input("\nYou: ").strip()

    if not text:
        continue

    if text == ":quit":
        print("Exiting.")
        break

    elif text == ":settings" or text == ":examples":
        show_settings()
        continue

    elif text == ":on":
        enabled = True
        print("Trigger enabled.")
        continue

    elif text == ":off":
        enabled = False
        print("Trigger disabled; inputs will not activate the action.")
        continue

    elif text.startswith(":threshold "):
        try:
            value = float(text.split(maxsplit=1)[1])
            if not 0 <= value <= 1:
                raise ValueError
            threshold = value
            print(f"Threshold updated to {threshold:.2f}")
        except ValueError:
            print("Use a number between 0 and 1.")
        continue

    elif text.startswith(":add "):
        example = text[5:].strip()
        if example:
            examples.append(example)
            prototype = build_prototype()
            print(f"Added example: {example}")
            print("Prototype rebuilt.")
        continue

    elif text.startswith(":remove "):
        try:
            index = int(text.split(maxsplit=1)[1]) - 1
            if not 0 <= index < len(examples):
                raise ValueError
            if len(examples) == 1:
                print("Keep at least one example.")
                continue
            removed = examples.pop(index)
            prototype = build_prototype()
            print(f"Removed: {removed}")
            print("Prototype rebuilt.")
        except ValueError:
            print("Use :remove followed by a valid example number.")
        continue

    vector = normalize(encoder.encode([text])[0])
    score = float(np.dot(vector, prototype))

    print(f"Similarity: {score:.3f} | Threshold: {threshold:.2f}")

    if not enabled:
        print("TRIGGER DISABLED — no action.")
    elif score >= threshold:
        print("🔔 TRIGGER FIRED")
        print("Demo action: set_temperature")

        choice = input("  [R]eturn tool result to model, [S]top, [N]o action? ").strip().lower()

        if choice == "r":
            print("Routing: tool result → model (integration point).")
        elif choice == "s":
            print("Routing: stop after action.")
        else:
            print("Action ignored.")
    else:
        print("No trigger — continue normal model flow.")