# %%
import json
import os
import argparse
import tqdm
import re

# %%
def main(args):

    # open file (for small files)
    with open(args.file, 'r') as f_in, open(f'{args.file.split('.json')[0]}_succ_traj.jsonl', 'w') as f_out:
        if '.jsonl' in args.file:
            for line in tqdm.tqdm(f_in):
                sample = json.loads(line)
                # successful samples only get processed
                if sample['test_result']['correct']:
                    processed = process_single_sample(sample)
                    f_out.write(json.dumps(processed) + "\n")


# %%
def process_single_sample(sample: dict) -> list[dict]:
    """
    Extract full trajectory for single sample
    """

    processed = []
    # go through every sample


    # globals
    # extract instance_id and history
    id = sample['instance_id']

    task = sample['test_result']['task']
    output = sample['test_result']['output']
    is_correct = sample['test_result']['correct']
    setup_cmds = sample['test_result']['setup_commands']
    tests = sample['test_result']['tests']

    # traj, task = _process_history(hist)
    hist = sample['history']
    try:
        traj = extract_full_trajectory(hist)
    except Exception as e:
        print(e)
        return None

    return  {
        "instance_id": id,
        'task': task,
        'answer': output,
        'is_correct': is_correct,
        "trajectory": traj,
        'setup_commands': setup_cmds,
        'tests': tests
    } 


def extract_full_trajectory(hist: dict) -> list[dict]:
    """
    Extracts trajectory from history object.

    Latest version has the raw response and prompt trajectory located in the
    absolute last object, within the `model_response` -> choices[0] ->
        -> provider_specific_fields -> raw_resp or raw_prompt
    """

    # history is a list of lists where the 2nd list consists of one or more dicts.
    # for ease of processing, we will flatten it first
    flat_hist = [substep for step in hist for substep in step]
    # filter non-action items
    flat_hist = [step for step in flat_hist if 'id' in step.keys()]

    last_action = flat_hist[-1] 
    traj = None  #init

    try:
        provider_data = last_action['tool_call_metadata']['model_response']['choices'][0]['provider_specific_fields']
        traj = provider_data['raw_prompt']['messages']  # full trajectory except for last LLM reply
        last_msg = provider_data['raw_resp']
        # BUG FIX: if more than 1 thinking block, extract only the last one + function call
        # seems to happen only in the last message
        if last_msg.count("<think>") > 2:
            last_msg = _fix_last_assistant_message(last_msg)
        traj.append({
            'role': 'assistant',
            'content': last_msg
        })
    except Exception as e:
        print(f'Error: {str(e)}')
        raise e
    
    return traj


def _fix_last_assistant_message(msg: str) -> str:
    """
    In OpenHands distilled data, for unknown reason, there are either 2 or 3 thinking
    blocks in the response. To fix, we extract only the last thinking block as well as
    the function call.
    """
    # if no bug found, no changes
    if msg.count('<think>') < 1:
        return msg
    
    match_think = re.findall(r"<think>.*?</think>", msg, flags=re.DOTALL)
    last_think = match_think[-1]
    match_func = re.search(f"<function=.*", msg, flags=re.DOTALL)
    func = match_func.group(0)
    return last_think + '\n' + func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill a dataset using an OpenAI-compatible model on shell tasks")
    parser.add_argument('--file', type=str, help='Path to output.json')
    args = parser.parse_args()

    main(args)

# %%
import glob
dir = '/shared_workspace_mfs/kirill/sft/OpenHands/evaluation/benchmarks/shell-tasks/distil-1500-promptv4-ds-31-official'

def get_completed_ids(dir: str):
    x = glob.glob(f'{dir}/*.json')
    x = list(map(lambda item: item.split('/')[-1].split('_')[-1].split('.json')[0], x))

    return x


def _process_history(hist: dict) -> list[dict]:
    """
    DEPRECATED, use extract_full_trajectory() instead
    Extracts trajectory from history object
    """
    # history is a list of lists where the 2nd list consists of one or more dicts.
    # for ease of processing, we will flatten it first
    flat_hist = [substep for step in hist for substep in step]
    traj = []
    task = None

    for step in flat_hist:
        
        # Extract data
        call = {}
        if 'id' not in step.keys():
            continue

        # always there
        call['id'] = step['id']
        call['source'] = step['source']
        call['message'] = step['message']

        if call['id'] == 1:
            # extract the task description
            # for prompt v1 version
            msg = call['message']
            task = msg.split('<issue_description>\n')[-1].split('\n</issue_description>\n')[0]
        
        # parsing special cases
        call['action'] = step.get('action', None)

        # running cmd call
        if "Running command" in call['message']:
            call['command'] = step['args']['command']
            call['thought'] = step['args']['thought']

        # observation call
        if "executed with exit code" in call['message']:
            call['cmd_output'] = step['content']
        
        # finishing call
        if call['action'] == "finish":
            call['thought'] = step['args']['final_thought']
        
        # append to the trajectory list
        traj.append(call)

    return traj, task