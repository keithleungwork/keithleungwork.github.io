# Path / files related

### Sub category :

[PDF skills](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/PDF%20skills%205aeec149822c4d7c827cf1073f234972.md)

[npz file](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/npz%20file%206ef3ea46fde447bc8e9fa5fec75e6486.md)

[json file](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/json%20file%20a3d18ba875af4ba19aaa5277ba9c78be.md)

### Path manipulation

#### join/resolve a path

```python
os.path.join("/somewhere/folder", "train.npz")
```

#### calculate size of every files in a dir (not subfolders)

```python
import os

model_path = "/somewhere"
sum(os.path.getsize(f.path) for f in os.scandir(model_path) if f.is_file())
```

[CSV file](Path%20files%20related%204ba1501b734c4d88a6d3d21eab820d15/CSV%20file%20e6500676a9c94f458abd847b626fdf5b.md)