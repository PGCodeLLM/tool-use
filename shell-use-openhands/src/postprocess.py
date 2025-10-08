"""
Post-processing scripts to go from raw OpenHands output.jsonl style files to SFT ready .jsonl
"""
# %%
import argparse
import json
import re

import pandas as pd
from tqdm import tqdm

from io_utils import read_jsonl, write_jsonl
from compress_multiturn_traj import flatten_multiturn
from extract_traj import process_single_sample, _fix_last_assistant_message
from extract_data import _preprocess_oh_shell_tasks_for_sft
from pg_format import merge_prompt_and_last_response

# %%
def main(args):
    assert args.fpath != None and '.jsonl' in args.fpath, 'No valid file path to a .jsonl file provided'

    if args.success_only:
        assert args.fpath != None and '.jsonl' in args.fpath, 'No valid file path to the output fil .jsonl file provided'
        print('--- Filtering for only successful samples ---')
        
        print("Loading the .jsonl (might take a while)...")
        df = pd.read_json(args.fpath, lines=True, orient='records')
        print(f'Pre-filter dataset size: {len(df)}...')
        df['correct'] = df['test_result'].apply(lambda x: x['correct'])
        df = df[df['correct'] == True]
        print(f'Post-filter dataset size: {len(df)}...')

        fname = args.output_file.split('.jsonl')[0] + f"_correct_{len(df)}s.jsonl"
        print(f'Saving to {fname}...')
        df.to_json(fname, orient='records', lines=True)
        print('Done!')
        exit(0) # exiting
    
    if args.extract_traj:
        assert args.output_file != None and '.jsonl' in args.output_file, 'No valid file path to the output file .jsonl file provided'

        print('--- Extracting trajectory and metadata from OpenHands logs ---')
        print("Loading the .jsonl (might take a while)...")
        fname = args.output_file.split('.jsonl')[0] + f".jsonl"  #TODO: add anything here?

        print('Extracting trajectory and metadata from OH logs sample-by-sample...')
        failed_count = 0
        with open(args.fpath, 'r') as f_in, open(fname, 'w') as f_out:
            for line in tqdm(f_in):
                sample = json.loads(line)
                processed = process_single_sample(sample)
                if processed == None:
                    failed_count += 1
                    continue
                f_out.write(json.dumps(processed) + "\n")

        print(f"ATTENTION: {failed_count} samples failed to extract trajectory! See error messages for details.")

        print('Done!')
        exit(0) # exiting

    if args.process_to_sft:
        print(f"--- Formatting data to be SFT ready (fast-thinking only: {args.fast_think}) ---")
        tags = f'fastthink' if args.fast_think else ''
        _ = _preprocess_oh_shell_tasks_for_sft(args.fpath, 
                                               save_to_disk=True, 
                                               fast_think=args.fast_think, 
                                               tags=tags)
        print('Done!')
        exit(0) # exiting

    if args.remove_failed_calls:
        print('---  Removes any interactions from trajectory that require OpenHands to reply with "Please continue working on..." propmt ---')
        
        print("Loading the .jsonl (might take a while)...")
        df = pd.read_json(args.fpath, lines=True, orient='records')
        
        print('Cleaning up trajectories...')
        traj_key = 'data' if 'data' in df.columns else 'trajectory'
        df[traj_key] = df[traj_key].apply(_remove_failed_calls)

        fname = args.fpath.split('.jsonl')[0] + f"_cleaned.jsonl"
        print(f'Saving to {fname}...')
        df.to_json(fname, orient='records', lines=True)

        print('Done!')
        exit(0) # exiting

    if args.flatten:
        print(f'---  Flattens multi-turn trajectory into a single-turn by simulating multi-turn in 1 turn. Compresssed everything into the user message (flat-token-type = {args.flat_token_type}). MUST BE used with NON-SFT ready data as code assumes there is `trajectory` key ---')
        _ = flatten_multiturn(input_jsonl_fpath=args.fpath, token_type=args.flat_token_type)
        print('Done!')
        exit(0) # exiting

    if args.convert_to_flat_pg:
        print(f'--- Flattens multi-turn and converts trajectory to PG format ---')
        
        print("Loading the .jsonl (might take a while)...")
        df = pd.read_json(args.fpath, lines=True, orient='records')

        print("Process every sample...")
        def _flat_pg_fn(traj):
            sys, usr, asst = merge_prompt_and_last_response(traj,
                                                            apply_pangu_template=True,
                                                            lastthink=args.lastthink)
            # convert few shot examples in prompt to pg as well (contain USER: and ASSISTANT: tags)
            usr_msg = usr['content']

            if args.no_few_shot:
                usr['content'] = _remove_few_shot_examples(usr_msg)

            return [sys, usr, asst]
        
        df['trajectory'] = df['trajectory'].apply(_flat_pg_fn)

        tag = '_nofewshot' if args.no_few_shot else ''
        if args.lastthink:
            tag += '_lastthink'
        fname = args.fpath.split('.jsonl')[0] + f"_flat_pg{tag}.jsonl"
        print(f'Saving to {fname}...')
        df.to_json(fname, orient='records', lines=True)

        print('Done!')
        exit(0) # exiting

    if args.fix_multi_think_blocks:
        print(f'--- Fix last assistnat message by extracting the last thinking block and function call. MUST BE done on non-SFT data as code required `trajectory` key! WARNING! This will overwrite the data! ---')
        
        assert 'trajectory' in df.columns, "The data must have a `trajectory` key, check that you're not using SFT-ready data!"

        print("Loading the .jsonl (might take a while)...")
        df = pd.read_json(args.fpath, lines=True, orient='records')

        # fix last message by extracting the last thinking block and
        # the function call
        def _fix_last_msg(traj: list[dict]):
            traj[-1]['content'] = _fix_last_assistant_message(traj[-1]['content'])
            return traj
        df['trajectory'] = df['trajectory'].apply(_fix_last_msg)

        # Saving
        # fname = args.fpath.split('.jsonl')[0] + f"_fixed.jsonl"
        print(f'Saving to {args.fpath}...')
        df.to_json(args.fpath, orient='records', lines=True)

        print('Done!')
        exit(0) # exiting

def _remove_few_shot_examples(text):
    END_OF_EXAMPLE = '--------------------- NEW TASK DESCRIPTION ---------------------'
    return END_OF_EXAMPLE + text.split(END_OF_EXAMPLE)[-1]


# %%
def _remove_failed_calls(traj: list[dict]) -> list[dict]:
    """
    TODO: refactor at some point to avoid two sequential loops, add metadata of how many calls were removed.

    Goes through the trajectory and removes interactions where the OpenHands agent
    replies with the "Please continue working on the task on..." (see below error_msg
    for full text). Removes the error call and one before it (the one that caused it)
    """

    error_msg = "Please continue working on the task on whatever approach you think is suitable.\nWhen you think you have solved the question, please use the finish tool and include your final answer in the message parameter of the finish tool.\nIMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP."

    new_traj = []
    idx_to_remove = []
    for idx, call in enumerate(traj):
        if error_msg in call['content']:
            idx_to_remove += [idx-1, idx]   # removing error and causing message

    # remove error calls
    for idx, call in enumerate(traj):
        if idx in idx_to_remove:
            continue
        else:
            new_traj.append(call)

    return new_traj 


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,
                        help='Path to .jsonl file')
    parser.add_argument('--output-file', type=str, default=None)
    
    # --- add additional post-processing arguments here ---
    parser.add_argument('--success-only', action='store_true', help='Filters to leave only successful/correct samples')
    
    # flattens multi-turn trajectory into simulated-muliturn by squashing everything into user message
    parser.add_argument('--flatten', action='store_true', 
                        help='Flatten multi-turn trajectory into single turn, as simulated multi-turn using a --flat-token-type as placeholder for roles.')
    parser.add_argument('--flat-token-type', type=str, default=None, 
                        help='For multi-turn flattening, which token type to use for user and assistant message; e.g. `ds-qwen3` will use "<｜User｜>: " for user tags. When not present, will use generic User and Assistant')

    # remove unsuccessful openhands interactions
    # where OH has to ask model to keep working (usually due to tool formatting issues)
    parser.add_argument('--remove-failed-calls', action='store_true',
                        help='remove unsuccessful openhands interactions where OH has to ask model to keep working (usually due to tool formatting issues)')
    
    # process raw OH trajectory into SFT format {benchmark_ids: .., data:..., meta_prompt:...}
    parser.add_argument('--extract_traj', action='store_true', 
                        help='Extract trajectory and metadata from raw OH logs; returns (id, task, output, is_correct, trajectory, setup_commands, tests)')
    
    # process to SFT ready format (To be refactored)
    parser.add_argument('--process-to-sft', action='store_true', 
                        help='Processs into SFT format (benchmark_ids:str, data:list[dict], meta_prompt:[str]), in a slow/fast think format depending on --fast-think argument')
    parser.add_argument('--fast-think', action='store_true', 
                        help='Keeps only fast thinking data (no <think> content). Defaults to false (has slow think)')

    # flatten and convert to PG format
    parser.add_argument('--convert-to-flat-pg', action='store_true', 
                        help='Flattens trajectory into simulated multiturn like --flatten, keeps only last assistant message thinking, and converts special tokens to PG format.')
    parser.add_argument('--lastthink', action='store_true', 
                        help='Only keeps thinking in the last assistant message (rest of trajectory has no thinking content)')
    parser.add_argument('--no-few-shot', action='store_true', 
                        help='Removes one-shot OpenHands example from initial user prompt')


    # BUG FIX: fixing OpenHands trajectories where the last assistant message 
    # has multiple thinking blocks (at least 2)
    parser.add_argument('--fix-multi-think-blocks', action='store_true', 
                        help='Fixing bug in OpenHands trajectories where the last assistant message has multiple thinking blocks. \nWARNING: this will overwrite the original data!\nMUST BE on non-SFT-ready data (has to contain `trajectory` key)')


    args = parser.parse_args()
    main(args)
    


