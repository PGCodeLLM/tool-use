"""Provided by Kishan"""
# %%
import os
import re
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Keep or remove?
PG_FORMAT_MAP = {
    "<think>": "[unused16]",
    "</think>": "[unused17]",
    "user": ""
}

def remove_think_blocks(text: str) -> str:
    """
    Remove all <think>...</think> blocks from the given text.
    """
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)

def extract_last_think_block(text: str):
    """
    Extract the last <think>...</think> block and the remaining answer.
    """
    match = re.search(r"(<think>.*?</think>)(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return "", text.strip()

def merge_prompt_and_last_response(messages, apply_pangu_template=False, lastthink=True):
    """
    Keep the first system message as-is.
    Merge all user and assistant messages (except the last assistant)
    into a single user message.
    Keep the last assistant message with its <think> block intact.
    Optionally apply Pangu-style formatting.
    """
    if not messages:
        return None, None, None

    system_msg = next((m for m in messages if m["role"] == "system"), None)
    if system_msg is None:
        system_msg = {"role": "system", "content": ""}

    compressed_lines = []
    first_user_encountered = False
    assistant_count = 0

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        if msg is system_msg:
            continue
        if i == len(messages) - 1 and role == "assistant":
            continue

        # Strip leading newlines if applying Pangu template
        if apply_pangu_template:
            content = content.lstrip("\n")

        if role == "assistant":
            assistant_count += 1
            clean_content = remove_think_blocks(content) if lastthink else content
            # if apply_pangu_template:
            #     line = f"/no_think[unused10][unused9]助手：[unused16][unused17]{clean_content.strip()}"
            # else:
            #     line = f"Assistant: {clean_content.strip()}"
            if apply_pangu_template:
                if lastthink:
                    line = f"/no_think[unused10][unused9]助手：[unused16][unused17]{clean_content.strip()}"
                else:
                    thinking, tool_call = extract_last_think_block(clean_content)
                    # Replace <think>\n with [unused16]
                    thinking = re.sub(r"<think>\s*\n*", "", thinking)

                    # Replace \n</think>\n\n with [unused17]
                    thinking = re.sub(r"\n*</think>\s*\n*", "", thinking)

                    # Strip any trailing newlines after replacement
                    thinking = thinking.rstrip("\n")
                    
                    line = f"[unused10][unused9]助手：[unused16]{thinking}[unused17]{tool_call.strip()}"
            else:
                # TODO: update to non-lastthink format
                line = f"Assistant: {clean_content.strip()}"
            compressed_lines.append(line)

        elif role == "user":
            if apply_pangu_template and not first_user_encountered:
                line = content.strip()
                first_user_encountered = True
            elif apply_pangu_template:
                line = f"[unused10][unused9]用户：{content.strip()}"
            else:
                line = f"User: {content.strip()}"
            compressed_lines.append(line)


    # user_msg = {"role": "user", "content": "\n".join(compressed_lines).strip()}
    user_msg = {"role": "user", "content": "".join(compressed_lines).strip()}
    # user_msg = messages[1] # temp for no trajectory

    last_msg = messages[-1]
    if last_msg["role"] != "assistant":
        return None, None, None

    think, answer = extract_last_think_block(last_msg["content"])
    final_assistant = {"role": "assistant", "content": f"{think}\n\n{answer}".strip()}
    if apply_pangu_template:
        # think = think.replace("<think>\n", "[unused16]").replace("\n</think>\n\n", "[unused17]")
        # Replace <think>\n with [unused16]
        think = re.sub(r"<think>\s*\n*", "[unused16]", think)

        # Replace \n</think>\n\n with [unused17]
        think = re.sub(r"\n*</think>\s*\n*", "[unused17]", think)

        # Strip any trailing newlines after replacement
        think = think.rstrip("\n")
    
        final_assistant = {"role": "assistant", "content": f"{think}{answer}".strip()}

    return system_msg, user_msg, final_assistant

# %%
def process_subseq_jsons(subseq_dir, output_dir, apply_pangu_template=False):
    """
    Process all JSON files in subseq_dir, transform them into the
    system + compressed user + last assistant format, and save to output_dir.
    Optionally apply Pangu-style formatting.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(subseq_dir)):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(subseq_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])
        if not messages or messages[-1]["role"] != "assistant":
            continue

        system_msg, user_msg, final_assistant = merge_prompt_and_last_response(messages, apply_pangu_template)
        if not system_msg or not user_msg or not final_assistant:
            continue

        output_data = {"messages": [system_msg, user_msg, final_assistant]}

        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved merged messages to: {output_dir}")




def get_all_json_data(path, output_path):
    """
    Scan all JSON files in a directory and collect metadata into a CSV.
    """
    analysis = []
    for file_name in os.listdir(path):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(path, file_name)
        traces = json.load(open(file_path, "r"))
        for idx in traces:
            trace = traces[idx]
            analysis.append(
                {
                    "file_name": file_name,
                    "repo": file_name.split("__")[0],
                    "trace_id": idx,
                    "length": len(trace["messages"]),
                    "stop_reason": trace["stop_reason"],
                    "test_status": trace["test_status"],
                    "test_error": trace["test_error"],
                    "resolved": trace["resolved"],
                }
            )

    df = pd.DataFrame(analysis)
    df.to_csv(output_path, index=False)
    return df


def analyze_json(path):
    """
    Calculate and print the success rate from the CSV produced by get_all_json_data().
    """
    df = pd.read_csv(path)
    success_rate = df[df["resolved"] == True].shape[0] / df.shape[0]
    print(f"Success rate: {success_rate:.2%}")
    return


def generate_subseq_data(data_dir, csv_path, output_dir):
    import pandas as pd

    df = pd.read_csv(csv_path)
    valid_df = df[(df["resolved"] == True) & (df["stop_reason"] == "Model finished")]

    output_splits_dir = os.path.join(output_dir, f"{os.path.basename(data_dir)}_subseq")
    merged_output_dir = os.path.join(output_dir, f"{os.path.basename(data_dir)}_merged")
    os.makedirs(output_splits_dir, exist_ok=True)
    os.makedirs(merged_output_dir, exist_ok=True)

    for index in tqdm(valid_df.index):
        file_name = valid_df.loc[index, "file_name"]
        file_path = os.path.join(data_dir, file_name)
        jd = json.load(open(file_path, "r"))

        trace_id = str(valid_df.loc[index, "trace_id"])
        trace = jd.get(trace_id)
        if (
            not trace
            or trace["stop_reason"] != "Model finished"
            or not trace["resolved"]
        ):
            continue

        message = trace["messages"]
        if message[-1]["role"] == "user":
            message = message[:-1]

        start = 0
        end = 0
        while end < len(message):
            if message[end]["role"] != "assistant":
                end += 1
                continue

            subseq = {"messages": message[start : end + 1]}
            subseq_filename = f"{file_name[:-5]}_{trace_id}_{start}_{end}.json"
            subseq_path = os.path.join(output_splits_dir, subseq_filename)

            with open(subseq_path, "w", encoding="utf-8") as f:
                json.dump(subseq, f, indent=4)

            # Now create merged prompt + final response
            prompt, response = merge_prompt_and_last_response(subseq["messages"])
            merged_data = {"prompt": prompt, "response": response}

            merged_path = os.path.join(merged_output_dir, subseq_filename)
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, indent=2)

            end += 1

    print(f"Saved subsequences to: {output_splits_dir}")
    print(f"Saved merged prompts to: {merged_output_dir}")


def clean_valid_jsons(valid_df, data_dir, cleaned_dir):
    os.makedirs(cleaned_dir, exist_ok=True)

    for index in tqdm(valid_df.index):
        file_name = valid_df.loc[index, "file_name"]
        trace_id = str(valid_df.loc[index, "trace_id"])
        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            jd = json.load(f)

        trace = jd.get(trace_id)
        if (
            not trace
            or trace["stop_reason"] != "Model finished"
            or not trace["resolved"]
        ):
            continue

        # Clean assistant messages
        cleaned_messages = []
        for msg in trace["messages"]:
            if msg.get("role") == "assistant":
                msg["content"] = remove_think_blocks(msg.get("content", ""))
            cleaned_messages.append(msg)

        # Remove final user message if it's <<< Finished >>>
        if cleaned_messages and cleaned_messages[-1].get("role") == "user":
            if "finished" in cleaned_messages[-1].get("content", "").lower():
                cleaned_messages = cleaned_messages[:-1]

        # Ensure last message is from assistant
        while cleaned_messages and cleaned_messages[-1].get("role") != "assistant":
            cleaned_messages.pop()

        # Save cleaned trace with only "messages"
        cleaned_trace = {"messages": cleaned_messages}
        cleaned_filename = f"{file_name[:-5]}_{trace_id}.json"
        cleaned_path = os.path.join(cleaned_dir, cleaned_filename)

        with open(cleaned_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_trace, f, indent=2)


def generate_nothink_data(data_dir, csv_path, cleaned_dir):
    df = pd.read_csv(csv_path)
    valid_df = df[(df["resolved"] == True) & (df["stop_reason"] == "Model finished")]

    clean_valid_jsons(valid_df, data_dir, cleaned_dir)


def generate_jsonl_from_json_dir(input_dir, output_path):
    total = 0
    cnt = defaultdict(list)
    with open(output_path, "w", encoding="utf-8") as outfile:
        for file in tqdm(os.listdir(input_dir)):
            sp = file.split("_")
            trace_id = sp[-3]
            name = "_".join(sp[:-3])
            if name in cnt and trace_id not in cnt[name] and len(cnt[name]) > 4:
                continue
            cnt[name].append(trace_id)
            # break
            round = int(sp[-1].replace(".json", "")) / 2
            # if "pydantic" in file and round >= 28:
            #     continue

            with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                # 如果是单个对象，直接写一行
                if isinstance(data, dict):
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    total += 1
    print(f"Total JSON objects written: {total}")
    print(len(cnt))
    return


if __name__ == "__main__":
    process_subseq_jsons(
         subseq_dir="/shared_workspace_mfs/kishan/data/SFT/mindforgeoh/deepSWE32B_rjsmp_swegymplus_all_think_subseq",

         output_dir="/shared_workspace_mfs/kishan/data/SFT/mindforgeoh/deepSWE32B_rjsmp_swegymplus_compressed_last_think_pangu_subseq",
         apply_pangu_template=True
    )

    subseq_output_dir = "/shared_workspace_mfs/kishan/data/SFT/mindforgeoh/deepSWE32B_rjsmp_swegymplus_compressed_last_think_pangu_subseq"
    generate_jsonl_from_json_dir(subseq_output_dir, "/shared_workspace_mfs/kishan/data/SFT/deepSWE32B_rjsmp_swegymplus_compressed_last_think_pangu_subseq.jsonl")
    print("Done")