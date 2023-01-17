# Import files - Best practice

> It is unbelievably annoying to do `import` in a python script.
> 

## Resource

- [https://blog.finxter.com/python-how-to-import-modules-from-another-folder](https://blog.finxter.com/python-how-to-import-modules-from-another-folder)
- [https://towardsdatascience.com/understanding-python-imports-init-py-and-pythonpath-once-and-for-all-4c5249ab6355](https://towardsdatascience.com/understanding-python-imports-init-py-and-pythonpath-once-and-for-all-4c5249ab6355)
- [https://peps.python.org/pep-0328/#rationale-for-relative-imports](https://peps.python.org/pep-0328/#rationale-for-relative-imports)
- [https://stackoverflow.com/questions/4209641/absolute-vs-explicit-relative-import-of-python-module](https://stackoverflow.com/questions/4209641/absolute-vs-explicit-relative-import-of-python-module)
- [https://stackoverflow.com/questions/16981921/relative-imports-in-python-3](https://stackoverflow.com/questions/16981921/relative-imports-in-python-3)

## Conclusion

- Use **absolute import** if possible, unless the path is too long.
- The entry point(which script will be executed) is critical

## Example:

Let’s say we have below structure:

![Untitled](Import%20files%20-%20Best%20practice%2042b1a5a69fe14275add56721ca3ff95c/Untitled.png)

We want to access CONST_A ,which is defined in `utils/module_share.py`, from `package_a/module_a.py`. However, we will execute `project_level.py`, which import `module_a.py`.

The main code is as below (note how the CONST_A is imported in different files)

```python
# src_project/project_level.py
from package_a.module_a import CONST_A

print(" CONST_A is: ", CONST_A)

# src_project/package_a/module_a.py
from utils.module_share import CONST_A

# src_project/utils/module_share.py
CONST_A = "123123"
```

However, if we import `project_level.py` in `root_level.py`, the import path everywhere will just fail.