"""Remove the 2.37 GiB temporary from the unchanged startup zero-weight predicate."""
from pathlib import Path
import sys

OLD = 'float(w.data.abs().max()) == 0.0'
NEW = 'not torch.any(w.data).item()'


def patch(path):
    path = Path(path)
    source = path.read_text(encoding='utf-8')
    if NEW in source and OLD not in source:
        return
    if source.count(OLD) != 1:
        raise RuntimeError('unsupported verify-head source; no file changed')
    path.write_text(source.replace(OLD, NEW), encoding='utf-8')


if __name__ == '__main__':
    patch(sys.argv[1])
