# json file

## Write to json

```python
import json
import os

json_file_content = {"test": 11111}
json_path = "/aaaa/bbb/aaa.json"
# In case folder not exist...
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, 'w') as outfile:
    json.dump(json_file_content, outfile, ensure_ascii=False, indent=4)
```

## Read from json

```python
import json

f_path = "/xxx/xxx/xxx/wdw.json"
with open(f_path) as f:
    json_obj = json.load(f)
```