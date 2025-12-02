# Excel Seat Checker

A Python tool to compare name lists with seat assignments in a special seat graph format.

## Features

- Reads Excel files with multiple sheets
- **Special seat graph format**: Automatically detects seat assignments by:
  - Finding cells with Chinese characters (names)
  - Checking the cell above for seat numbers
  - Recognizing seat formats: `1-001S` (floor-seat) or `加座1` (additional seat)
  - Ignoring merged/combined cells
- Sheet names represent building numbers
- Finds people in name list who don't have seats
- Finds seats that don't have corresponding people in the name list
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

### With Custom Options

```bash
# Specify name column in name list
python excel_seat_checker.py name_list.xlsx seat_graph.xlsx --name-column "Full Name"

# Adjust scan limits for large files
python excel_seat_checker.py name_list.xlsx seat_graph.xlsx --max-rows 2000 --max-cols 300

# Specify custom output file name (default: seat_assignment_output.xlsx)
python excel_seat_checker.py name_list.xlsx seat_graph.xlsx --output my_results.xlsx
```

## Excel File Format

### Name List File
- **Only first sheet is read**
- First row contains headers (e.g., "工号, 姓名, 入职日期, 三级部门, 四级部门")
- Name column is auto-detected as "姓名" (Chinese for "name"), or use `--name-column` to specify
- All columns are preserved in the output
- Example:
  ```
  工号 | 姓名 | 入职日期 | 三级部门 | 四级部门
  001  | 张三 | 2020-01-01 | 技术部 | 开发组
  002  | 李四 | 2020-02-01 | 技术部 | 测试组
  003  | 王五 | 2020-03-01 | 产品部 | 设计组
  ```

### Seat Graph File (Special Format)
- **Sheet names**: Building numbers (e.g., "Building 1", "Building 2")
- **Layout**: Each seat has 2 adjacent cells (top and bottom)
  - **Top cell**: Contains seat number (format: `1-001S` or `加座1`)
  - **Bottom cell**: Contains person name (Chinese characters)
- **Seat number formats**:
  - Standard: `1-001S` (floor number - seat number)
  - Additional: `加座1` (additional seat with no number)
- **Merged cells**: Automatically ignored
- Example layout:
  ```
  | 1-001S | 1-002S | 1-003S |
  |  张三  |  李四  |  王五  |
  | 加座1  |        | 2-001S |
  |  赵六  |        |  孙七  |
  ```

## How It Works

1. **Name Detection**: Scans cells for Chinese characters (Unicode range \u4e00-\u9fff)
2. **Seat Number Detection**: Checks the cell above the name cell for:
   - Pattern `\d+-\d+S` (e.g., "1-001S", "2-123S")
   - Pattern `加座\d+` (e.g., "加座1", "加座2")
3. **Merged Cell Handling**: Skips cells that are part of merged cell ranges
4. **Building Identification**: Uses sheet names as building identifiers

## Output

The script generates an **Excel file** with:
1. **All original columns** from the name list file
2. **New "Seat" column** containing seat assignments in format: `Building-SeatNumber` (e.g., `Building 1-1-001S`)
3. **All rows from name list** with seat information filled in where available
4. **Additional rows** for unmatched seats (empty seats or seats with people not in name list) - these rows have empty values for name list columns, only seat info is filled

### Console Output

The script also prints analysis results to console:
1. **People without seats**: Names from the name list that don't appear in any seat assignment
2. **Empty seats**: Seats that either have no name assigned or have names not in the name list

Example console output:
```
1. People in name list WITHOUT seats: 2
   - 张三
   - 李四

2. Seats WITHOUT corresponding people in name list: 3
   - Building 'Building 1', Seat '1-001S': (empty)
   - Building 'Building 1', Seat '加座1': 赵六
   - Building 'Building 2', Seat '2-001S': (empty)
```

### Excel Output Format

The output Excel file contains:
- **All original columns from name list**: Preserved in original order (e.g., 工号, 姓名, 入职日期, 三级部门, 四级部门, etc.)
- **Seat column**: Appended at the end with seat assignment in format `Building-SeatNumber` (e.g., `Building 1-1-001S`)
- **Rows with seat info**: Original name list rows with seat column filled
- **Rows without seat info**: Original name list rows with empty seat column (people without seats)
- **Unmatched seat rows**: New rows with only seat column filled, other columns empty (empty seats or seats with people not in name list)

Example output structure:
```
工号 | 姓名 | 入职日期 | 三级部门 | 四级部门 | Seat
001  | 张三 | 2020-01-01 | 技术部 | 开发组 | Building 1-1-001S
002  | 李四 | 2020-02-01 | 技术部 | 测试组 | Building 1-1-002S
003  | 王五 | 2020-03-01 | 产品部 | 设计组 | 
     |      |           |         |         | Building 1-加座1
```

## Programmatic Usage

You can also import and use the functions in your own code:

```python
from excel_seat_checker import read_name_list, read_seat_graph, check_seats

name_list = read_name_list("name_list.xlsx")
seat_graph = read_seat_graph("seat_graph.xlsx", max_rows=1000, max_cols=200)
people_without_seats, empty_seats = check_seats(name_list, seat_graph)
```

## Notes

- Names are normalized (lowercase, trimmed) for comparison
- Empty cells are ignored
- The script handles multiple sheets (buildings) automatically
- Chinese character detection uses Unicode range \u4e00-\u9fff
- Merged cells are automatically detected and skipped
- Default scan limits: 1000 rows × 200 columns per sheet (adjustable)
