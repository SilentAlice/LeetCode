# LeetCode Solutions

Solutions to LeetCode problems in multiple programming languages.

## Structure

- `c/` - C implementations of LeetCode problems
- `python/` - Python implementations of LeetCode problems

## C Solutions

The `c/` directory contains C implementations. Each problem file follows the naming convention `lc{problem_number}.c`.

## Python Solutions

The `python/` directory contains Python implementations with:
- `problems/` - Individual problem solutions
- `utils/` - Helper utilities and common data structures (ListNode, TreeNode)
- `tests/` - Unit tests for solutions

See `python/README.md` for more details on the Python project structure.

## Usage

### C Solutions
```bash
cd c
gcc lc1.c -o lc1
./lc1
```

### Python Solutions
```bash
cd python
pip install -r requirements.txt
python problems/lc1.py
pytest tests/
```
