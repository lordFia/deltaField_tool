import os
import shlex
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"

STORE_DIR = "latent_store"
COMMAND_DIR = os.path.join(STORE_DIR, "commands")

os.makedirs(COMMAND_DIR, exist_ok=True)


# ============================================================
# Load model
# ============================================================

print()
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded.")
print()


# ============================================================
# Current inference state
# ============================================================

last_prompt = None
last_input_ids = None
last_hidden_states = None


# ============================================================
# Utility
# ============================================================

def token_text(token_id):
    return tokenizer.decode([int(token_id)])


def token_info(index):
    if last_input_ids is None:
        return None

    if index < 0 or index >= len(last_input_ids):
        return None

    token_id = last_input_ids[index]

    return {
        "index": index,
        "id": int(token_id),
        "text": token_text(token_id),
    }


def find_token(text, occurrence=0):
    """
    Find a token by decoded text.

    Example:
        find_token("B")

    Note:
        BPE tokenizers may encode words with spaces/newline markers,
        so token matching is based on the decoded token.
    """

    if last_input_ids is None:
        return None

    matches = []

    for i, token_id in enumerate(last_input_ids):
        decoded = token_text(token_id)

        if decoded == text:
            matches.append(i)

    if occurrence >= len(matches):
        return None

    return matches[occurrence]


# ============================================================
# Inference
# ============================================================

def infer(text, display_tokens=False):
    global last_prompt
    global last_input_ids
    global last_hidden_states

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(
            **inputs,
            output_hidden_states=True
        )

    # hidden_states contains:
    #
    #   [embedding output,
    #    layer 1,
    #    layer 2,
    #    ...
    #    final layer]
    #
    # Each tensor:
    #
    #   [batch, sequence, hidden_dimension]

    hidden_states = torch.stack(
        [
            h[0].detach().cpu()
            for h in output.hidden_states
        ]
    )

    last_prompt = text
    last_input_ids = inputs["input_ids"][0].detach().cpu()
    last_hidden_states = hidden_states

    print()
    print(f"Prompt: {text!r}")
    print(
        f"Tokens: {last_hidden_states.shape[1]}    "
        f"Layers: {last_hidden_states.shape[0] - 1}    "
        f"Hidden size: {last_hidden_states.shape[2]}"
    )

    if display_tokens:
        show_tokens()

    print()

    return output


# ============================================================
# Display tokens
# ============================================================

def show_tokens():
    if last_input_ids is None:
        print("No inference has been run.")
        return

    print()
    print("TOKEN STREAM")
    print("-" * 60)

    for i, token_id in enumerate(last_input_ids):
        text = token_text(token_id)

        print(
            f"{i:4d}   "
            f"id={int(token_id):6d}   "
            f"{text!r}"
        )

    print("-" * 60)
    print()


# ============================================================
# Resolve token argument
# ============================================================

def resolve_token(argument):
    """
    TOKEN can be:

        5
        "B"

    If a number is supplied, it is interpreted as an index.

    If text is supplied, the first matching token is used.
    """

    if last_input_ids is None:
        print("No inference has been run.")
        return None

    # Try integer index first.
    try:
        index = int(argument)

        if 0 <= index < len(last_input_ids):
            return index

        print(f"Token index {index} is out of range.")
        return None

    except ValueError:
        pass

    # Otherwise search decoded token.
    index = find_token(argument)

    if index is None:
        print(
            f"Could not find token {argument!r}. "
            f"Use 'tokens' to inspect tokenization."
        )
        return None

    return index


# ============================================================
# Trace token
# ============================================================

def get_token_trajectory(token_index):
    if last_hidden_states is None:
        return None

    if token_index < 0:
        return None

    if token_index >= last_hidden_states.shape[1]:
        return None

    return last_hidden_states[:, token_index, :]


def show_trace(token_index):
    trajectory = get_token_trajectory(token_index)

    if trajectory is None:
        print("Invalid token.")
        return

    info = token_info(token_index)

    print()
    print("TOKEN TRAJECTORY")
    print("-" * 75)

    print(
        f"Token index : {info['index']}"
    )

    print(
        f"Token       : {info['text']!r}"
    )

    print(
        f"Token ID    : {info['id']}"
    )

    print()

    print(
        f"{'Layer':>8} "
        f"{'||h||':>15} "
        f"{'||Δh||':>15} "
        f"{'Δ cosine':>15}"
    )

    print("-" * 75)

    for layer in range(len(trajectory)):

        h = trajectory[layer]

        h_norm = torch.norm(h).item()

        if layer == 0:
            delta_norm = 0.0
            cosine = 1.0

        else:
            delta = trajectory[layer] - trajectory[layer - 1]

            delta_norm = torch.norm(delta).item()

            cosine = torch.nn.functional.cosine_similarity(
                trajectory[layer - 1].unsqueeze(0),
                trajectory[layer].unsqueeze(0)
            ).item()

        print(
            f"{layer:8d} "
            f"{h_norm:15.6f} "
            f"{delta_norm:15.6f} "
            f"{cosine:15.6f}"
        )

    print("-" * 75)
    print()


# ============================================================
# Build transformation trace
# ============================================================

def build_operator_trace(
    trajectory,
    start_layer,
    end_layer
):
    """
    Convert a trajectory:

        h_start
        h_start+1
        ...
        h_end

    into the sequence:

        Δ_start
        Δ_start+1
        ...
        Δ_end-1

    where:

        Δ_l = h_(l+1) - h_l
    """

    selected = trajectory[
        start_layer:end_layer + 1
    ]

    operators = []

    for i in range(len(selected) - 1):
        delta = selected[i + 1] - selected[i]
        operators.append(delta)

    return torch.stack(operators)


# ============================================================
# Record command
# ============================================================

def record_command(
    name,
    text,
    token_argument,
    start_layer,
    end_layer
):

    try:
        start_layer = int(start_layer)
        end_layer = int(end_layer)

    except ValueError:
        print("START and END must be integers.")
        return

    if start_layer < 0:
        print("START must be >= 0.")
        return

    if end_layer <= start_layer:
        print("END must be greater than START.")
        return

    # Run source inference.
    infer(text)

    token_index = resolve_token(token_argument)

    if token_index is None:
        return

    trajectory = get_token_trajectory(token_index)

    max_layer = trajectory.shape[0] - 1

    if end_layer > max_layer:
        print(
            f"END layer {end_layer} exceeds "
            f"maximum layer {max_layer}."
        )
        return

    token = token_info(token_index)

    # Record the actual transformation trace.
    operators = build_operator_trace(
        trajectory,
        start_layer,
        end_layer
    )

    # Also retain the original trajectory segment.
    trajectory_segment = trajectory[
        start_layer:end_layer + 1
    ].clone()

    command = {
        "name": name,

        "model": MODEL_NAME,

        "source_text": text,

        "token_index": token_index,

        "token_id": token["id"],

        "token_text": token["text"],

        "start_layer": start_layer,

        "end_layer": end_layer,

        "trajectory": trajectory_segment,

        "operators": operators,
    }

    path = os.path.join(
        COMMAND_DIR,
        name + ".pt"
    )

    torch.save(command, path)

    print()
    print("COMMAND RECORDED")
    print("-" * 60)

    print(f"Name        : {name}")
    print(f"Source      : {text!r}")
    print(f"Token       : {token['text']!r}")
    print(f"Token index : {token_index}")
    print(
        f"Layers      : "
        f"{start_layer} -> {end_layer}"
    )
    print(
        f"Operators   : "
        f"{len(operators)}"
    )
    print(f"Saved       : {path}")

    print("-" * 60)
    print()


# ============================================================
# List commands
# ============================================================

def list_commands():

    files = sorted(
        f
        for f in os.listdir(COMMAND_DIR)
        if f.endswith(".pt")
    )

    if not files:
        print("No commands recorded.")
        return

    print()
    print("RECORDED COMMANDS")
    print("-" * 75)

    for filename in files:

        path = os.path.join(
            COMMAND_DIR,
            filename
        )

        try:
            command = torch.load(
                path,
                map_location="cpu",
                weights_only=False
            )

            print(
                f"{command['name']:20s} "
                f"token={command['token_text']!r:8s} "
                f"layers="
                f"{command['start_layer']}"
                f"->"
                f"{command['end_layer']}"
            )

        except Exception as e:

            print(
                f"{filename[:-3]:20s} "
                f"[ERROR: {e}]"
            )

    print("-" * 75)
    print()


# ============================================================
# Show command
# ============================================================

def show_command(name):

    path = os.path.join(
        COMMAND_DIR,
        name + ".pt"
    )

    if not os.path.exists(path):
        print(
            f"Command {name!r} does not exist."
        )
        return

    command = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print()
    print("COMMAND")
    print("-" * 60)

    print(f"Name        : {command['name']}")
    print(f"Model       : {command['model']}")
    print(
        f"Source      : "
        f"{command['source_text']!r}"
    )
    print(
        f"Token       : "
        f"{command['token_text']!r}"
    )
    print(
        f"Token index : "
        f"{command['token_index']}"
    )
    print(
        f"Layers      : "
        f"{command['start_layer']}"
        f" -> "
        f"{command['end_layer']}"
    )
    print(
        f"Operators   : "
        f"{len(command['operators'])}"
    )

    print()
    print("OPERATOR MAGNITUDES")
    print("-" * 60)

    for i, operator in enumerate(
        command["operators"]
    ):

        layer = (
            command["start_layer"] + i
        )

        magnitude = torch.norm(
            operator
        ).item()

        print(
            f"{layer:4d} -> {layer + 1:4d}   "
            f"{magnitude:.8f}"
        )

    print("-" * 60)
    print()


# ============================================================
# Call command
# ============================================================

def call_command(
    name,
    text,
    token_argument
):

    path = os.path.join(
        COMMAND_DIR,
        name + ".pt"
    )

    if not os.path.exists(path):
        print(
            f"Command {name!r} does not exist."
        )
        return

    command = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    # Run new inference.
    infer(text)

    token_index = resolve_token(
        token_argument
    )

    if token_index is None:
        return

    new_trajectory = get_token_trajectory(
        token_index
    )

    start_layer = command["start_layer"]
    end_layer = command["end_layer"]

    if end_layer >= new_trajectory.shape[0]:
        print(
            "Command was recorded for a deeper "
            "model than the current trajectory."
        )
        return

    operators = command["operators"]

    token = token_info(token_index)

    print()
    print("CALLING COMMAND")
    print("-" * 75)

    print(f"Command     : {name}")
    print(f"Input       : {text!r}")
    print(
        f"Token       : "
        f"{token['text']!r}"
    )
    print(
        f"Token index : "
        f"{token_index}"
    )
    print(
        f"Layers      : "
        f"{start_layer} -> {end_layer}"
    )

    print()
    print(
        "Applying recorded transformation "
        "trace..."
    )
    print()

    # Start with the new token's actual
    # hidden state at the command's start layer.
    replayed_state = (
        new_trajectory[start_layer]
        .clone()
    )

    print(
        f"{'Layer':>12} "
        f"{'Replay ||h||':>16} "
        f"{'Actual ||h||':>16} "
        f"{'Cosine':>12} "
        f"{'Error':>16}"
    )

    print("-" * 80)

    for i, operator in enumerate(operators):

        layer = start_layer + i
        next_layer = layer + 1

        # Apply the recorded transformation.
        replayed_state = (
            replayed_state + operator
        )

        actual_state = (
            new_trajectory[next_layer]
        )

        cosine = (
            torch.nn.functional
            .cosine_similarity(
                replayed_state.unsqueeze(0),
                actual_state.unsqueeze(0)
            )
            .item()
        )

        error = torch.norm(
            replayed_state - actual_state
        ).item()

        replay_norm = torch.norm(
            replayed_state
        ).item()

        actual_norm = torch.norm(
            actual_state
        ).item()

        print(
            f"{layer:5d} -> {next_layer:<5d} "
            f"{replay_norm:16.6f} "
            f"{actual_norm:16.6f} "
            f"{cosine:12.6f} "
            f"{error:16.6f}"
        )

    print("-" * 80)

    # Final comparison.
    actual_final = (
        new_trajectory[end_layer]
    )

    final_cosine = (
        torch.nn.functional
        .cosine_similarity(
            replayed_state.unsqueeze(0),
            actual_final.unsqueeze(0)
        )
        .item()
    )

    final_error = torch.norm(
        replayed_state - actual_final
    ).item()

    print()
    print("FINAL COMPARISON")
    print("-" * 60)

    print(
        f"Cosine similarity : "
        f"{final_cosine:.8f}"
    )

    print(
        f"Euclidean error   : "
        f"{final_error:.8f}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "The transformer itself was NOT modified."
    )

    print(
        "The recorded operator was replayed "
        "mathematically on the new token's "
        "hidden state."
    )

    print("-" * 60)
    print()


# ============================================================
# Help
# ============================================================

def help_text():

    print("""
LATENT COMMAND SYSTEM
======================

INFERENCE
---------

  infer "TEXT"

      Run inference and retain hidden states.

      Example:

        infer "What is the lowercase for B?"

  tokens

      Display tokenization of the most recent prompt.


TOKEN TRACE
-----------

  trace TOKEN

      Trace one token through all transformer layers.

      TOKEN can be either a token index:

        trace 5

      or the decoded token:

        trace "B"


RECORD
------

  record NAME "TEXT" TOKEN START END

      Record the layer-by-layer transformation trace
      of TOKEN.

      Example:

        record LOWERCASE "A" "A" 2 8

      This stores:

        Δ₂ = h₃ - h₂
        Δ₃ = h₄ - h₃
        ...
        Δ₇ = h₈ - h₇

      under:

        LOWERCASE


CALL
----

  call NAME "TEXT" TOKEN

      Apply the recorded transformation sequence to
      TOKEN in a new prompt.

      Example:

        call LOWERCASE "B" "B"

      The system starts from B's hidden state at
      the recorded start layer and applies the
      LOWERCASE operators.


COMMAND MANAGEMENT
------------------

  list

      List recorded commands.

  show NAME

      Show details and operator magnitudes.


GENERAL
-------

  help

      Show this help.

  exit

      Quit.


IMPORTANT
---------

This version records and mathematically replays
hidden-state transformations.

It does NOT yet inject the modified hidden state
back into the transformer.

That will be the next experimental layer.
""")


# ============================================================
# Main interactive shell
# ============================================================

while True:

    try:
        line = input("latent> ").strip()

    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not line:
        continue

    try:
        parts = shlex.split(line)

    except ValueError as e:
        print(f"Parse error: {e}")
        continue

    command = parts[0].lower()


    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if command in ("exit", "quit"):
        break


    # --------------------------------------------------------
    # Help
    # --------------------------------------------------------

    elif command == "help":
        help_text()


    # --------------------------------------------------------
    # Infer
    # --------------------------------------------------------

    elif command == "infer":

        if len(parts) != 2:
            print(
                'Usage: infer "TEXT"'
            )
            continue

        infer(
            parts[1],
            display_tokens=True
        )


    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    elif command == "tokens":

        show_tokens()


    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    elif command == "trace":

        if len(parts) != 2:
            print(
                'Usage: trace TOKEN'
            )
            continue

        index = resolve_token(
            parts[1]
        )

        if index is not None:
            show_trace(index)


    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    elif command == "record":

        if len(parts) != 6:

            print(
                'Usage: record NAME "TEXT" TOKEN START END'
            )

            print(
                'Example:'
            )

            print(
                '  record LOWERCASE "A" "A" 2 8'
            )

            continue

        name = parts[1]
        text = parts[2]
        token = parts[3]
        start = parts[4]
        end = parts[5]

        record_command(
            name,
            text,
            token,
            start,
            end
        )


    # --------------------------------------------------------
    # Call
    # ------------------------------------------