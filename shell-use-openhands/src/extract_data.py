# %% Init
import pandas as pd
import glob
import os
import numpy as np
from datasets import load_dataset
import json
import re

# DATA_DIR = "/shared_workspace_mfs/kirill/sft/tool_call_repeat_failure/sft_data/shell"

# %% Extract reasoning from <think> tokens
def _extract_fast_thinking(x: dict) -> dict:

    og_string = x[-1]['content']
    temp = og_string.split('<think>')[-1]
    temp = temp.split('</think>')
    x[-1]['content'] = "".join(temp)
    return x

# %%
# DATA_PATH = '/shared_workspace_mfs/kirill/sft/sft-pipeline/sft-runs/data'
# raw_data_fpath = f'{DATA_PATH}/full/oh-shell-tasks-1225s-mt-distill-ds-31.jsonl'

def _remove_thinking(traj: list[dict]) -> list[dict]:
    print('Removing thinking data...')
    # removes slow thinking from trajectory
    for idx in range(len(traj)):
        if traj[idx]['role'] == 'assistant':
            traj[idx]['content'] = re.sub(r'<think>.*?</think>', "", traj[idx]['content'], flags=re.DOTALL).lstrip('\n')
    return traj

def _preprocess_oh_shell_tasks_for_sft(fpath: str, save_to_disk: bool = False, fast_think: bool = False, tags: str = '') -> pd.DataFrame:
    # read jsonl
    df = pd.read_json(fpath, lines=True, orient='records')
    
    # removes slow thinking from trajectory, surrounded by <think></think> tags             
    if fast_think:
        df['trajectory'] = df['trajectory'].apply(lambda traj: _remove_thinking(traj))

    # generate expected format for sft
    df['meta_prompt'] = df['trajectory'].apply(lambda x: [x[0]['content']])
    df['data'] = df['trajectory'].apply(lambda x: x[1:])
    df['benchmark_task_id'] = df['instance_id']

    # only keep required columns
    df = df[['benchmark_task_id', 'meta_prompt', 'data']]

    # save to disk
    if save_to_disk:
        fname = fpath.split('.jsonl')[0]
        if tags:
            fname += f"_{tags}_sft_rdy.jsonl"
        else:
            fname += f"_sft_rdy.jsonl"
        df.to_json(fname, orient='records', lines=True)

    return df


# %% Preprocess 
def _process_shelllm_v3_simple_data(dir: str, thinking: bool = True, save_to_disk: bool = False, tags: str = '') -> pd.DataFrame:

    dataset_name = dir.split('/')[-1]
    dataset = load_dataset(dir)

    df = dataset['train'].to_pandas()
    print(f'Total samples: {len(df)}')

    # filter: 
    # leave only successful trajectories
    df = df[df['evaluation'].apply(lambda x: x['success_condition_passed'] == True)]
    print(f'Leaving only successful trajectories: {len(df)}')

    # take only samples with single turn
    df = df[df['trajectory'].apply(lambda x: len(x) == 1)]
    print(f'Leaving only single-turn samples: {len(df)}')

    # Extracts the shell command and thinking from the trajectory
    def _format_output(traj: list[dict]) -> str:
        d = traj[-1]
        if thinking:
            msg = {
                    'role': 'assistant',
                    'content': f'<think>{d["thought"]}</think>\n```\n{d["action"]}\n```'
                }
        else:
            msg = {
                    'role': 'assistant',
                    'content': f'```\n{d["action"]}\n```'
                }
        return msg
    df['output'] = df['trajectory'].apply(_format_output)
    df['input'] = df['task'].apply(lambda x: {'role': 'user', 'content': x})
    
    # create columns for SFT-ready dataset format
    df['benchmark_task_id'] = df.index
    df['meta_prompt'] = np.empty((len(df), 0)).tolist()
    df['data'] = df.apply(lambda x: [x['input'], x['output']], axis=1)

    df = df[['benchmark_task_id', 'meta_prompt', 'data']]

    if save_to_disk:
        if thinking:
            fname = f"{DATA_DIR}/{dataset_name}_thinking_{len(df)}s_{tags}.jsonl"
        else:
            fname = f"{DATA_DIR}/{dataset_name}_nonthink_{len(df)}s_{tags}.jsonl"
        df.to_json(fname, orient='records', lines=True)

    return df

# %%
def _convert_shell_to_openhands_format(fdir: str, thinking: bool = False, save_to_disk: bool = False) -> pd.DataFrame:
    df = pd.read_json(fdir, lines=True, orient='records')

    def _convert_fn(data: list[dict]) -> list[dict]:
        if thinking:
            thinking_prompt, shell_cmd = data[1]['content'].split('```\n')
        else:
            shell_cmd = data[1]['content'].split('```\n')
        shell_cmd = shell_cmd.split('\n```')[0]
        if thinking:
            data[1]['content'] = thinking_prompt + f"\n<function=execute_bash>\n<parameter=command>{shell_cmd}</parameter>\n</function>"
        else:
            data[1]['content'] = f"\n<function=execute_bash>\n<parameter=command>{shell_cmd}</parameter>\n</function>"
        return data

    df['data'] = df['data'].apply(_convert_fn)

    if save_to_disk:
        fname = fdir.split('.jsonl')[0]
        fpath = f"{fname}_openhands.jsonl"

        df.to_json(fpath, orient='records', lines=True)

    return df

# %%
def _train_test_split(fpath: str, train_size: float = 0.8) -> None:

    df = pd.read_json(fpath, lines=True, orient='records')

    # shuffle, then sample train and test sets
    df = df.sample(frac=1.0)
    train_df = df.sample(frac=train_size, random_state=123)
    test_df = df.drop(train_df.index)

    splits = fpath.split('/')
    fname = splits[-1]
    path = "/".join(splits[:-1])
    train_df.to_json(f"{path}/train{len(train_df)}_{fname}", orient='records', lines=True)
    test_df.to_json(f"{path}/test{len(test_df)}_{fname}", orient='records', lines=True)

# %%
import re
def _extract_bash_cmds(bash_string:str):
    """
    Extracts bash commands, separated by &&
    """
    commands = re.findall(r"^\s*(\S+)|(?:&&|\||;)\s*(\S+)", bash_string)
    commands = [cmd for tup in commands for cmd in tup if cmd]
    # return commands
    if "python" in bash_string:
        if len(commands) > 1:
            return commands[0:2]
        else:
            return [commands[0]]
    else:
        return commands

def _analyze_tool_content(fpath: str):
    # Note: must be the non-thinking dataset
    df = pd.read_json(fpath, lines=True, orient='records')
    df['cmd'] = df['data'].apply(lambda x: _extract_bash_cmds(x[-1]['content'].split('```\n')[-1].split('\n```')[0]))

    # plot top 30 shell tools used in descending order
    cmds = pd.DataFrame(df['cmd'].sum())
    cmds['cmd'].value_counts().nlargest(30).plot(kind='bar')
# %%
