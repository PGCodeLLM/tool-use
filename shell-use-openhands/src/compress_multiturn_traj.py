"""
Compresses multi-turn trajectory into a simulated-multi-turn
by compressing user and assistant messages into a single
user command, and outputs the last assistant message as
the only assistant message. Done such that Llama-Factory
uses system prompt and entire multi-turn chat for conditioning
when predicting the last asisstant message

e.g. 
from:   U -> A -> U -> ... -> U -> A
to      U = 'User: xxx, Assistant: xxx ....', A: <last_assistant_message>

"""

import json
import argparse
from tqdm import tqdm
import copy

DS_R1_0528_QWEN3_SPECIAL_TOKENS = {
    "assistant": "<｜Assistant｜>: ",
    "user": "<｜User｜>: "
}

GENERIC_TOKENS = {
    "user": "User: ",
    "assistant": "Assistant: "
}

def flatten_multiturn(input_jsonl_fpath: str, token_type: str = None) -> list[dict]:
    # Select which tokens should be used as substitution
    # Add other ones if needed
    match args.token_type:
        case 'ds-qwen3':
            token_map = DS_R1_0528_QWEN3_SPECIAL_TOKENS
        case _:
            token_map = GENERIC_TOKENS

    # read the jsonl with multi-turn data
    with open(args.input_jsonl, 'r') as f:
        og_data = [json.loads(line) for line in f]

    # go through every sample
    new_data = []
    for sample in tqdm(og_data):
        traj = sample['trajectory']

        # compress the trajectory
        new_traj = [
            {}, # placeholder for system prompt ChatML message
            {
                'role': 'user',
                'content': ''
            },
            {} # placeholder for last assistant message (for loss calcualtion)
        ]
        new_traj[0] = traj[0]   # assumed that first msg is always system prompt
        new_traj[-1] = traj[-1] # assumed last message is from assistant (checked true from OH distillation)

        # compress the intermediate steps
        for step in traj[1:-2]:
            role = step['role']
            new_traj[1]['content'] = new_traj[1]['content'] + token_map[role] + step['content']+ '\n'
        
        # save to new_data list
        new_dict = copy.deepcopy(sample)
        new_dict['trajectory'] = new_traj
        new_data.append(new_dict)

    # save to new jsonl
    new_fname = args.input_jsonl.split('.jsonl')[0] + '_flat_multiturn.jsonl'
    print(f'Saving to {new_fname}...')
    with open(new_fname, 'w') as f:
        for line in new_data:
            f.write(json.dumps(line) + '\n')

    return new_data

if __name__  == '__main__':
    # parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-jsonl', type=str)
    # parser.add_argument('--output-jsonl', type=str)
    parser.add_argument('--token-type', type=str, default=None)
    args = parser.parse_args()
    flatten_multiturn(args.input_jsonl, args.token_type)