import json

def read_jsonl(fpath: str) -> list[dict]:
    with open(fpath, "r") as file:
        data = [json.loads(line) for line in file]
    return data


def write_jsonl(fpath:str, data: list[dict]):
    with open(fpath, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')


def read_json(fpath: str) -> dict:
    with open(fpath, "r") as file:
        data = json.load(file)
    return data


def write_json(fpath:str, data:dict):
    with open(fpath, 'w') as outfile:
        json.dump(data, outfile)

def read_log(fpath:str) -> str:
    log = []
    with open(fpath, 'r') as f:
        for line in f:
            x = line.strip()
            if x:
                log.append(x)

    return log