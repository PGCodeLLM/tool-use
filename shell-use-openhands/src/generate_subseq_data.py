# %%
import json
import argparse
from tqdm import tqdm
import copy

def generate_subseq_data(messages: list[dict]) -> list[list[dict]]:
    """
    Generate "trajectory ladder" samples from a single trajectory.
    e.g. A-B-C-D-E-F ==> (A-B, A-B-C, A-B-C-D, A-B-C-D-E,...)
    """
    new_samples = []
    end_idx = 0
    while end_idx < len(messages):
        if messages[end_idx]['role'] != 'assistant':
            end_idx += 1
            continue

        # append a new subsequence to data
        new_samples.append(messages[:end_idx + 1])
        end_idx += 1
    
    return new_samples

# %%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate 'trajectory ladder' samples by subsequencing a single multi-turn trajectory; done on the non-SFT formatted data as code expects a 'trajectory'")
    parser.add_argument('--fpath', type=str, help='path to JSONL file')
    args = parser.parse_args()

    # load the original data
    with open(args.fpath, 'r') as f:
        data = [json.loads(line.rstrip('\n')) for line in f]

    # generate subsequences for the provided data
    processed = []
    for instance in tqdm(data):
        traj = instance['trajectory']
        new_samples = generate_subseq_data(traj)
        
        # save subsequenced samples
        for sid, subseq in enumerate(new_samples):
            instance_copy = copy.deepcopy(instance)
            instance_copy['trajectory'] = subseq
            instance_copy['instance_id'] = instance_copy['instance_id'] + f'_s{sid + 1}'

            processed.append(instance_copy)
    
    # Save to file
    print(f'Samples before subsequencing: {len(data)}')
    print(f'Samples before subsequencing: {len(processed)}')

    fname = args.fpath.split('.jsonl')[0] + f'_subseq{len(processed)}.jsonl'
    with open(fname, 'w') as f:
        for sample in processed:
            json.dump(sample, f)
            f.write('\n')
    print('Done!')

