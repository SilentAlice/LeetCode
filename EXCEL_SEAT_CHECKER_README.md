# Excel Seat Checker

A Python tool to compare name lists with seat assignments across multiple Excel sheets.

## Features

- Reads Excel files with multiple sheets
- Automatically detects column names (seat numbers, names)
- Finds people in name list who don't have seats
- Finds seats that don't have corresponding people in the name list
- Supports custom column names
- Can export results to CSV

## Installation

```bash
pip install -r excel_seat_checker_requirements.txt
```

## Usage

### Basic Usage

```bash
python excel_seat_checker.py name_list.xlsx seat_graph.xlsx
```

### With Custom Column Names

If your Excel files use specific column names:

```bash
python excel_seat_checker.py name_list.xlsx seat_graph.xlsx \
    --name-column "Full Name" \
    --seat-column "Seat Number" \
    --seat-name-column "Assigned Person"
```

### Export Results to CSV

```bash
python excel_seat_checker.py name_list.xlsx seat_graph.xlsx --output results.csv
```

## Excel File Format

### Name List File
- Can have multiple sheets
- Contains names in one column (first column by default, or specify with `--name-column`)
- Example:
  ```
  Name
  John Doe
  Jane Smith
  Bob Johnson
  ```

### Seat Graph File
- Can have multiple sheets
- Each sheet contains seat assignments
- Should have columns for:
  - Seat numbers (auto-detected by looking for "seat", "num", "number" keywords)
  - Person names (auto-detected by looking for "name", "person", "people" keywords)
- Example:
  ```
  Seat Number | Person Name
  A1          | John Doe
  A2          | Jane Smith
  B1          | (empty)
  ```

## Output

The script provides:
1. **People without seats**: Names from the name list that don't appear in any seat assignment
2. **Empty seats**: Seats that either have no name assigned or have names not in the name list

## Programmatic Usage

You can also import and use the functions in your own code:

```python
from excel_seat_checker import read_name_list, read_seat_graph, check_seats

name_list = read_name_list("name_list.xlsx")
seat_graph = read_seat_graph("seat_graph.xlsx")
people_without_seats, empty_seats = check_seats(name_list, seat_graph)
```

## Notes

- Names are normalized (lowercase, trimmed) for comparison
- Empty cells are ignored
- The script handles multiple sheets automatically
