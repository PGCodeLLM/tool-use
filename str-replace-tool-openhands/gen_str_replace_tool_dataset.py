"""
Generates dataset for SFT on OpenHands' str_replace_editor tool usage from distilled traces
"""
# %% imports
import glob
import json
import re
from tqdm import tqdm
import os

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
        cmd = cmatch[0] if pmatch else None

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
if __name__ == "__main__":

    SOURCE_DATA_PATH = "/shared_workspace_mfs/datasets/mindforgeoh-rjsmp/deepSWE32B_rjsmp_swegymplus"
    file_paths = glob.glob(f"{SOURCE_DATA_PATH}/*")

    SAVE_DIR = "/shared_workspace_mfs/kirill/tool-use/str-replace-tool-openhands/data"

    for fpath in tqdm(file_paths):
        _ = process_instance_json(fpath)
        
# %%
