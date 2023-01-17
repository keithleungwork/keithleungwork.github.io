# Image lib

---

## Skimage

### Must-read - in/out datatypes

Skimage accepts various Input datatypes (or range, e.g. -1 to 1 or 0 to 255)

- [https://scikit-image.org/docs/dev/user_guide/data_types.html](https://scikit-image.org/docs/dev/user_guide/data_types.html)

### Convert output to other ranges:

```python
from skimage.util import img_as_float
image = np.arange(0, 50, 10, dtype=np.uint8)
print(image.astype(float)) # These float values are out of range.

print(img_as_float(image))
```

### Draw an image with skimage:

```python
from skimage import io
p = "/ssss/xxxx.png"
image = io.imshow(p)
```