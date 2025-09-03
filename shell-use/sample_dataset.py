#!/usr/bin/env python3
import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Common bash tool whitelist
COMMON_BASH_CMDS = {
    "find", "grep", "sort", "wc", "awk", "sed", "head", "tail", "uniq", "cut",
    "ls", "cat", "chmod", "chown", "tar", "gzip", "gunzip", "bzip2", "xz",
    "du", "df", "mkdir", "rmdir", "mv", "cp", "rm", "echo", "tee", "xargs",
    "ps", "kill", "top", "less", "more", "tr", "date", "whoami", "pwd",
    "touch", "stat", "env", "export", "history"
}

# Original data files
ORIGINAL_FILES = [
    "shellm-V3-simple-unix_thinking_0818_7452s_single_turn_openhands.jsonl",
    "shellm-V3-simple-unix_thinking_0818_7452s_single_turn.jsonl"
]

# Define dataset groups
EASY_DATASETS = [
    "bashtrain",
    "rlvr-bash-unedited",
    "shell_alpaca",
    "shell_medium",
    "text_to_bash",
    "toolmaxx-shell"
]

HARD_DATASETS = [
    "qwen3-235b-shell-tasks",
    "rlvr-bash-edited",
    "shell-tasks",
    "shellm-V3-simple-unix",
    "V3-shell-format"
]

THINKING_DATASETS = [
    "qwen3-235b-shell-tasks",
    "shellm-V3-simple-unix",
    "V3-shell-format"
]

def _extract_bash_cmds(bash_string: str):
    """Extract bash commands"""
    commands = re.findall(r"^\s*(\S+)|(?:&&|\||;)\s*(\S+)", bash_string)
    commands = [cmd for tup in commands for cmd in tup if cmd]
    return [c for c in commands if c in COMMON_BASH_CMDS]

def _extract_from_direct(content: str):
    """Extract commands from direct format"""
    if "```" in content:
        try:
            return _extract_bash_cmds(content.split("```\n")[-1].split("\n```")[0])
        except Exception:
            return _extract_bash_cmds(content)
    return _extract_bash_cmds(content)

def _extract_from_openhand(content: str):
    """Extract commands from openhand format"""
    match = re.search(r"<parameter=command>(.*?)</parameter>\s*</function>", content, re.S)
    if match:
        return _extract_bash_cmds(match.group(1).strip())
    return _extract_bash_cmds(content)

def extract_assistant_cmd(turns):
    """Extract commands from assistant's content depending on format"""
    for x in turns:
        if x.get("role") == "assistant":
            content = x.get("content", "")
            if "</parameter>" in content and "</function>" in content:
                return _extract_from_openhand(content)
            else:
                return _extract_from_direct(content)
    return []

def plot_top30_commands(all_cmds, out_path):
    """Plot top 30 common bash tools and print frequencies"""
    cmds = pd.DataFrame(all_cmds, columns=["cmd"])
    if cmds.empty:
        print("⚠️ No commands found, skip plotting")
        return

    top30 = cmds["cmd"].value_counts().nlargest(30)

    # Print to console
    print("\n=== Top 30 Common Bash Tools ===")
    for cmd, freq in top30.items():
        print(f"{cmd:10s} : {freq}")
    print("================================\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    ax = top30.plot(kind="bar")

    plt.title("Top 30 common bash tools")
    plt.ylabel("Frequency")
    plt.xticks(rotation=60)

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom", fontsize=8, rotation=90
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved figure to {out_path}")

def sample_dataset(input_dir, orig_size, new_size,
                   easy_ratio, thinking_ratio,
                   direct_ratio=0.5):
    tool_ratio = 1 - direct_ratio
    all_records = []
    all_cmds = []

    # === Load original data ===
    original_records = []
    for f in ORIGINAL_FILES:
        fpath = os.path.join(input_dir, f)
        if os.path.exists(fpath):
            df = pd.read_json(fpath, lines=True, orient="records")
            df["_source"] = "original"
            df["_format"] = "original"
            original_records.extend(df.to_dict(orient="records"))

    # === Load new data (direct.jsonl / tool.jsonl separately) ===
    for dataset in os.listdir(input_dir):
        dataset_dir = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for fmt in ["direct.jsonl", "tool.jsonl"]:
            fpath = os.path.join(dataset_dir, fmt)
            if not os.path.exists(fpath):
                continue
            df = pd.read_json(fpath, lines=True, orient="records")
            for record in df.to_dict(orient="records"):
                record["_dataset"] = dataset
                record["_format"] = fmt.replace(".jsonl", "")
                record["_source"] = "sampled"
                all_records.append(record)

    df_new = pd.DataFrame(all_records)
    df_orig = pd.DataFrame(original_records)

    sampled = []

    # === Sample old data ===
    if not df_orig.empty:
        sampled.extend(
            df_orig.sample(n=min(orig_size, len(df_orig)), random_state=42).to_dict(orient="records")
        )

    # === Split new data into easy / hard / thinking ===
    if not df_new.empty:
        df_easy = df_new[df_new["_dataset"].isin(EASY_DATASETS)]
        df_hard = df_new[df_new["_dataset"].isin(HARD_DATASETS)]
        df_thinking = df_new[df_new["_dataset"].isin(THINKING_DATASETS)]
        df_hard_no_thinking = df_hard[~df_hard["_dataset"].isin(THINKING_DATASETS)]

        # Calculate sample sizes
        easy_count = int(new_size * easy_ratio)
        hard_count = new_size - easy_count
        hard_thinking_count = int(hard_count * thinking_ratio)
        hard_no_thinking_count = hard_count - hard_thinking_count

        # === 分配 direct / tool 比例 ===
        def sample_with_format(df, total_count):
            n_direct = int(total_count * direct_ratio)
            n_tool = total_count - n_direct
            df_direct = df[df["_format"] == "direct"]
            df_tool = df[df["_format"] == "tool"]

            chosen = []
            if not df_direct.empty and n_direct > 0:
                chosen.extend(
                    df_direct.sample(n=min(n_direct, len(df_direct)), random_state=42).to_dict(orient="records")
                )
            if not df_tool.empty and n_tool > 0:
                chosen.extend(
                    df_tool.sample(n=min(n_tool, len(df_tool)), random_state=42).to_dict(orient="records")
                )
            return chosen

        if easy_count > 0:
            sampled.extend(sample_with_format(df_easy, easy_count))
        if hard_thinking_count > 0:
            sampled.extend(sample_with_format(df_thinking, hard_thinking_count))
        if hard_no_thinking_count > 0:
            sampled.extend(sample_with_format(df_hard_no_thinking, hard_no_thinking_count))

    # === Extract commands for statistics ===
    for rec in sampled:
        if "data" in rec:
            cmds = extract_assistant_cmd(rec["data"])
            for c in cmds:
                all_cmds.append([c])

    # === Save sampling results ===
    out_dir = "sample_dataset"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"orig{orig_size}_new{new_size}_easy{easy_ratio}_think{thinking_ratio}_direct{direct_ratio}.jsonl"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Sampling finished: {len(sampled)} samples -> {out_path}")

    # === Save command distribution figure ===
    plot_top30_commands(all_cmds, out_path.replace(".jsonl", "_common_bash_top30.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample SFT dataset and plot bash tool distribution")
    parser.add_argument("--input_dir", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--orig_size", type=int, required=True, help="Number of samples from original data")
    parser.add_argument("--new_size", type=int, required=True, help="Number of samples from new datasets")
    parser.add_argument("--easy_ratio", type=float, required=True, help="Easy ratio (0~1, only for new data)")
    parser.add_argument("--thinking_ratio", type=float, required=True, help="Thinking ratio (0~1, only for hard data)")
    parser.add_argument("--direct_ratio", type=float, default=0.5, help="Ratio of direct samples in new data (tool ratio = 1 - direct_ratio)")

    args = parser.parse_args()

    sample_dataset(
        args.input_dir,
        args.orig_size,
        args.new_size,
        args.easy_ratio,
        args.thinking_ratio,
        args.direct_ratio
    )
