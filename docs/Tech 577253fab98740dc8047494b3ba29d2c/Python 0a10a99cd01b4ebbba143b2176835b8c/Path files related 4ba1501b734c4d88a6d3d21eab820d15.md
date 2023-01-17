# Path / files related

---

## Search within folder :

### Via `glob`

```python
import glob
import os
character_json_paths = glob.glob(os.path.join(DIR_JSON, "*.json"))
```

### Via simple `os.listdir`

```python
from pathlib import Path
from typing import List
import os

def search_files_of_type(
    path: str, suffix: str = ".json",
    exclude_files_name: List[str] = [],
    exclude_ignore_suffix: bool = True
) -> list:
    # Remove suffix if True
    if exclude_ignore_suffix and len(exclude_files_name) > 0:
        new_excl = [Path(v).with_suffix("") for v in exclude_files_name]
        # listdir ONLY list the direct level, NOT all nested path
        return [v for v in os.listdir(path) if v.endswith(suffix) and Path(v).with_suffix("") not in new_excl]
    else:
        return [v for v in os.listdir(path) if v.endswith(suffix) and v not in exclude_files_name]
```

---

## Path manipulation

### File suffix(i.e. extension)

```python
v = "/ada/wdwdw.txt"
filename_no_suffix = str(Path(v).with_suffix(""))
# /ada/wdwdw
```

### join/resolve a path :

```python
import os
os.path.join("/somewhere/folder", "train.npz")

# or
from pathlib import Path
Path("/somewhere").join("train.npz")
```

### calculate size of every files in a dir (not subfolders) :

```python
import os

model_path = "/somewhere"
sum(os.path.getsize(f.path) for f in os.scandir(model_path) if f.is_file())
```

---

## Temp folder/file

### Via `tempfile` :

```python

with tempfile.TemporaryFile()as fp:
    fp.write(b'Hello world!')
    fp.seek(0)
    fp.read()

## file is now closed and removed

with tempfile.TemporaryDirectory() as path:
    images_from_path = convert_from_path(
				"/home/user/example.pdf", 
				output_folder=path)
		# do something....
```

---

## General - Text File

### Read file:

Get all lines into a `list`

```python
pdf_list = []
with open("./pdf_list.log") as f:
    pdf_list = f.readlines()
```

---

## Sub category :

[PDF skills](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/PDF%20skills%205aeec149822c4d7c827cf1073f234972.md)

[npz file](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/npz%20file%206ef3ea46fde447bc8e9fa5fec75e6486.md)

[json file](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/json%20file%20a3d18ba875af4ba19aaa5277ba9c78be.md)

[CSV file](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/CSV%20file%20e6500676a9c94f458abd847b626fdf5b.md)

[H5PY](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/H5PY%207268064783ad44a4a47c8c83cb50323a.md)