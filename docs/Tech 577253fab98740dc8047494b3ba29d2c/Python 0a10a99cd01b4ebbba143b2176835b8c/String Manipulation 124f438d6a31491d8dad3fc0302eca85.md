# String Manipulation

---

## Regex

### Common pattern

```python
r'_\d+$' # e.g. xxxxx_12
```

### Replace pattern with xxx:

```python
import re
target_text = "hahahahha_111"
re.sub(r'_\d+$', '', target_text)
# hahahahha
```

### Search pattern:

```python
import re
target_text = "hahahah_111"
m = re.search(r'_\d+$',target_text)
if m is not None:
    get_str = m.group()
    print(get_str)
else:
    print("no match.")
# output: _111
```