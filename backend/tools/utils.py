import json
import os
import shutil
import json
import re
import pandas as pd


def extract_numbers(filename):
    pattern = r"(\w+)_(\d+)-\d+"
    match = re.match(pattern, filename)
    if match:
        name = str(match.group(1))
        start = int(match.group(2))
        return name, start
    else:
        return filename, 0


def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def read_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def write_json(data, file_name, save_path="./", mode="r+"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, f"{file_name}.json")

    if not os.path.exists(full_path):
        with open(full_path, 'w') as file:
            pass
    elif mode == "w":
        with open(full_path, 'w') as file:
            json.dump(data, file, indent=4)
        return

    with open(full_path, 'r+') as file:
        for key, value in data.items():
            if is_file_empty(full_path):
                file.write('{}')
                file.seek(0, os.SEEK_END)
                file.seek(file.tell() - 1, os.SEEK_SET)
                file.write('\n')
                file.write(json.dumps(key, indent=4))
                file.write(': ')
                file.write(json.dumps(value, indent=4))
                file.write('\n')
                file.write('}')
                continue

            file.seek(0, os.SEEK_END)
            file.seek(file.tell() - 2, os.SEEK_SET)
            file.write(',')
            file.write('\n')
            file.write(json.dumps(key, indent=4))
            file.write(': ')
            file.write(json.dumps(value, indent=4))
            file.write('\n')
            file.write('}')
    return

def clear_file(defile_name, save_path="res"):
    if not os.path.exists(save_path):
        print(f"The path {save_path} does not exist!")
        return

    for root, dirs, files in os.walk(save_path):
        for dir_name in dirs:
            if dir_name == defile_name:
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Folder '{defile_name}' has been deleted: {dir_path}")
                continue

        for file in files:
            file_name = file.split('.')[0]
            if defile_name == file_name:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"{file_path} has been deleted.")
                
if __name__ == "__main__":
    test = True
    if not test:
        exit(0)

    data4 = {
        'frame':
        360,
        'court': [[671, 471], [1251, 471], [629, 678], [1293, 674], [540, 987],
                  [1370, 987]]
    }

    write_json(data4, "demo")

    import pandas as pd

    data = pd.DataFrame(pd.read_json("demo.json"))
    print(data['frame'])