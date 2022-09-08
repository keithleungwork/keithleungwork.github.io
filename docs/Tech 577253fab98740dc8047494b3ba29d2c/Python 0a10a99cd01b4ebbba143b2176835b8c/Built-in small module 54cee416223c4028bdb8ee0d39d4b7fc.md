# Built-in small module

## globals

- returns a dictionary with all the global variables and symbols for the current program.

```python
print(globals())

""" for example : 
{'In': ['', 'globals()'],
 'Out': {},
 '_': '',
 '__': '',
 '___': '',
 '__builtin__': <module 'builtins' (built-in)>,
 '__builtins__': <module 'builtins' (built-in)>,
 '__name__': '__main__',
 '_dh': ['/home/repl'],
 '_i': '',
 '_i1': 'globals()',
 '_ih': ['', 'globals()'],
 '_ii': '',
 '_iii': '',
 '_oh': {},
 '_sh': <module 'IPython.core.shadowns' from '/usr/local/lib/python3.5/dist-packages/IPython/core/shadowns.py'>,
 'exit': <IPython.core.autocall.ExitAutocall at 0x7fbc60ca6c50>,
 'get_ipython': <bound method InteractiveShell.get_ipython of <IPython.core.interactiveshell.InteractiveShell object at 0x7fbc6478ee48>>,
 'quit': <IPython.core.autocall.ExitAutocall at 0x7fbc60ca6c50>}
"""
```

## _ _ all _ _ :

- _ _ all _ _ affects the `from <module> import *` ONLY
    - i.e. `from <module> import <something>` still works for all var

```python
# foo.py
__all__ = ['bar', 'baz']

waz = 5
bar = 10
def baz(): return 'baz'

# main.py
from foo import *

print(bar)
print(baz)

# The following will trigger an exception, as "waz" is not exported by the module
print(waz)
```