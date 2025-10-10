"""
Generates dataset for SFT on OpenHands' str_replace_editor tool usage from distilled traces
"""
# %% imports
import glob
import json
import re
from tqdm import tqdm
import os
import pandas as pd

# %% extract tool interactions and their context
def extract_str_replace_tool_interactions(messages: list[dict]) -> list[dict]:

    new_data = []

    for idx in range(len(messages)):

        # ignore system prompt
        if idx == 0: 
            continue

        text = messages[idx]['content']

        # check if current message has str_replace_editor tool call
        # otherwise, ignore
        tool_check_str = "<function=str_replace_editor>"
        if not tool_check_str in text:
            continue

        # if tool call is the last message, quit because we don't have a tool call result
        if idx == len(messages) - 1:
            break

        # extract the context (all messages before) and the result (one message after)
        context = messages[:idx]
        result = messages[idx+1]

        # extract attributes from the tool call, helps later processing
        # file path
        pmatch = re.findall(r".*?<parameter=path>(.*?)</parameter>.*?", text, re.DOTALL)
        path = pmatch[0] if pmatch else None

        # command
        cmatch = re.findall(r".*?<parameter=command>(.*?)</parameter>.*?", text, re.DOTALL)
        cmd = cmatch[0] if cmatch else None

        # tool call result
        if all([x in result['content'] for x in ['ERROR:', 'Exit code:', 'Execution output of']]):
            call_status = 'fail'
        else:
            call_status = 'success'
        
        # append sample to global dict
        new_data.append({
            'context': context,
            'tool_call': messages[idx],
            'result': result,
            'path': path,
            'command': cmd,
            'status': call_status
        })

        # update idx to point to next message after this interaction
        idx += 2

    return new_data

# %% single json processing
def process_instance_json(json_fpath: str, save_path: str = None) -> list[dict]:
    """
    Extract training samples from OpenHands distillation data
    Input data format:
    {
        '0': {
            'messages': ...
            'stop_reason': ...
            'test_status': ...
            'test_error': ...
            'patch': ...
            'resolved': ...
        }, ,,,
    }
    """
    with open(json_fpath, 'r') as f:
        data = json.load(f)

    instance_id = json_fpath.split('/')[-1].split('.json')[0]

    # go through every attempt
    new_data = []
    for att_k, att_v in tqdm(data.items()):
        
        # extract training samples
        new_samples = extract_str_replace_tool_interactions(messages=att_v['messages'])

        # Save every interaction as separate sample and
        # add instance_id and other attributes
        for idx, sample in enumerate(new_samples):
            d = {
                'instance_id': instance_id,
                'sample_id': instance_id + f"_att{att_k}_s{idx}"
            } | sample
            new_data.append(d)

    if save_path:
        fname = json_fpath.rsplit('/',1)[-1].split('.json')[0] + '.jsonl'
        fpath = os.path.join(save_path, fname)
        print(f'..saving to {fpath}')
        with open(fpath, 'w') as f:
            for data in new_data:
                f.write(json.dumps(data) + '\n')

    return new_data

# %%
def convert_str_replace_to_sft_ready(jsonl_fpath: str, convert_to_pg: bool):
    """
    TODO: Refactor! Thrown together very quickly
    """
    with open(jsonl_fpath, 'r') as f:
        data = [json.loads(line.rstrip('\n')) for line in f]

    new_data = []

    for sample in tqdm(data):
        if sample['command'] != 'str_replace' and sample['status'] != 'success':
            continue

        # convert to pg format simulated-multiturn and last think
        trajectory = sample['context'] + [sample['tool_call']]
        sys_msg, usr_msg, final_assistant = merge_prompt_and_last_response(trajectory,
                                                                           apply_pangu_template=True,
                                                                           lastthink=True)

        # HACK: workaround for the remaining <think> tag
        if '<think>' in final_assistant['content']:
            final_assistant['content'] = final_assistant['content'].replace('<think>', '')

        # new_data.append({
        #     'benchmark_task_id': data['sample_id'],
        #     'meta_propmt': sys_msg['content'],
        #     'data': [usr_msg, final_assistant] 
        # })

        with open('/shared_workspace_mfs/kirill/tool-use/str-replace-tool-openhands/data/str_replace_tool_call_lastthink_multiturn.jsonl', 'a') as f:
            f.write(json.dumps({
                        'benchmark_task_id': sample['sample_id'],
                        'meta_prompt': sys_msg['content'],
                        'data': [usr_msg, final_assistant] 
                    }) + '\n')


    # df = pd.read_json(jsonl_fpath, lines=True, orient='records')
    # df = df[(df['status'] == 'success') and (df['command'] == 'str_replace')]

    # # df['meta_prompt'] = df['context'].apply(lambda x: x[0]['content'])
    # df['traj'] = df.apply(lambda row: row['context'] + row['tool_call'])

    # def _lastthink_simulated_multiturn(messages):
    #     sys_msg, usr_msg, final_assistant = merge_prompt_and_last_response(messages,
    #                                                                        apply_pangu_template=True,
    #                                                                        lastthink=True)
    #     return sys_msg, [usr_msg, final_assistant]
    # df[['meta_prompt', 'data']] = df['traj'].apply(_lastthink_simulated_multiturn(), result_type='expand')

    # df = df


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
    # TODO: For some reason, the assistant message still has one <think> tag after all this processing
    think, answer = extract_last_think_block(last_msg["content"])
    final_assistant = {"role": "assistant", "content": f"{think}\n\n{answer}".strip()}
    if apply_pangu_template:
        # think = think.replace("<think>\n", "[unused16]").replace("\n</think>\n\n", "[unused17]")
        # Replace <think>\n with [unused16]
        think = re.sub(r"\s*\n*<think>\s*\n*", "[unused16]", think)

        # Replace \n</think>\n\n with [unused17]
        think = re.sub(r"\s*\n*</think>\s*\n*", "[unused17]", think)

        # Strip any trailing newlines after replacement
        think = think.rstrip("\n")
    
        final_assistant = {"role": "assistant", "content": f"{think}{answer}".strip()}

    return system_msg, user_msg, final_assistant


# %%
if __name__ == "__main__":

    # SOURCE_DATA_PATH = "/shared_workspace_mfs/datasets/mindforgeoh-rjsmp/deepSWE32B_rjsmp_swegymplus"
    # file_paths = glob.glob(f"{SOURCE_DATA_PATH}/*")

    # SAVE_DIR = "/shared_workspace_mfs/kirill/tool-use/str-replace-tool-openhands/data/deepSWE32B_rjsmp_swegymplus"

    # for fpath in tqdm(file_paths):
    #     _ = process_instance_json(fpath, save_path=SAVE_DIR)

    # convert to pg fake-multiturn last think
    SOURCE_DATA_PATH = "/shared_workspace_mfs/kirill/tool-use/str-replace-tool-openhands/data/deepSWE32B_rjsmp_swegymplus"
    file_paths = glob.glob(f"{SOURCE_DATA_PATH}/*")

    # TODO: to refactor later
    for fpath in tqdm(file_paths):
        convert_str_replace_to_sft_ready(fpath, True)



# %%
