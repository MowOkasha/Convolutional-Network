# Data Folder Notes

Place your dataset under `data/raw` if you are using custom image folders.

Expected format:

```text
data/raw/
  train/
    class_1/*.jpg
    class_2/*.jpg
  val/
    class_1/*.jpg
    class_2/*.jpg
  test/
    class_1/*.jpg
    class_2/*.jpg
```

Notes:
- `test` is optional in code, but highly recommended for final reporting.
- Class folder names become your class labels automatically.
- Keep class names stable across train/val/test.
