#!/usr/bin/env python3
"""
Excel Seat Checker
Reads two Excel files:
1. Name list - contains people names
2. Seat graph - contains seat numbers and people names across multiple sheets

Checks:
- People in name list who don't have seats
- Seats that don't have corresponding people in the name list
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
import argparse


def normalize_name(name):
    """Normalize name for comparison (strip whitespace, convert to lowercase)"""
    if pd.isna(name):
        return None
    return str(name).strip().lower()


def read_name_list(file_path: str, name_column: str = None) -> Set[str]:
    """
    Read name list from Excel file.
    
    Args:
        file_path: Path to the name list Excel file
        name_column: Column name containing names. If None, uses first column.
    
    Returns:
        Set of normalized names
    """
    try:
        # Read all sheets and combine names
        excel_file = pd.ExcelFile(file_path)
        all_names = set()
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                continue
            
            # Use specified column or first column
            if name_column and name_column in df.columns:
                col = name_column
            else:
                col = df.columns[0]
            
            # Extract names
            names = df[col].dropna()
            normalized_names = {normalize_name(name) for name in names if normalize_name(name)}
            all_names.update(normalized_names)
            
            print(f"  Sheet '{sheet_name}': Found {len(normalized_names)} names")
        
        return all_names
    
    except Exception as e:
        print(f"Error reading name list file: {e}")
        sys.exit(1)


def read_seat_graph(file_path: str, seat_column: str = None, name_column: str = None) -> Dict[str, Dict[str, str]]:
    """
    Read seat graph from Excel file with multiple sheets.
    
    Args:
        file_path: Path to the seat graph Excel file
        seat_column: Column name containing seat numbers. If None, auto-detect.
        name_column: Column name containing names. If None, auto-detect.
    
    Returns:
        Dictionary: {sheet_name: {seat_num: person_name}}
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        seat_data = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            if df.empty:
                print(f"  Sheet '{sheet_name}': Empty, skipping")
                continue
            
            # Auto-detect columns if not specified
            if seat_column and seat_column in df.columns:
                seat_col = seat_column
            else:
                # Try to find seat column (look for 'seat', 'seat_num', etc.)
                seat_col = None
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'seat' in col_lower or 'num' in col_lower or 'number' in col_lower:
                        seat_col = col
                        break
                if seat_col is None:
                    seat_col = df.columns[0]  # Default to first column
            
            if name_column and name_column in df.columns:
                name_col = name_column
            else:
                # Try to find name column
                name_col = None
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'name' in col_lower or 'person' in col_lower or 'people' in col_lower:
                        name_col = col
                        break
                if name_col is None:
                    # Use second column if seat is first, otherwise first
                    name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            # Extract seat-number pairs
            sheet_seats = {}
            for _, row in df.iterrows():
                seat = row[seat_col]
                name = row[name_col]
                
                if pd.isna(seat):
                    continue
                
                seat_str = str(seat).strip()
                name_normalized = normalize_name(name)
                
                if seat_str:
                    sheet_seats[seat_str] = name_normalized
            
            seat_data[sheet_name] = sheet_seats
            print(f"  Sheet '{sheet_name}': Found {len(sheet_seats)} seats")
        
        return seat_data
    
    except Exception as e:
        print(f"Error reading seat graph file: {e}")
        sys.exit(1)


def check_seats(name_list: Set[str], seat_graph: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Check for missing seats and empty seats.
    
    Args:
        name_list: Set of normalized names from name list
        seat_graph: Dictionary of {sheet_name: {seat_num: person_name}}
    
    Returns:
        Tuple of (people_without_seats, empty_seats)
        empty_seats is list of (sheet_name, seat_num) tuples
    """
    # Collect all names that have seats
    names_with_seats = set()
    all_seats = []
    
    for sheet_name, seats in seat_graph.items():
        for seat_num, person_name in seats.items():
            all_seats.append((sheet_name, seat_num, person_name))
            if person_name:
                names_with_seats.add(person_name)
    
    # Find people in name list without seats
    people_without_seats = list(name_list - names_with_seats)
    
    # Find seats without corresponding people in name list
    empty_seats = []
    for sheet_name, seat_num, person_name in all_seats:
        if not person_name or person_name not in name_list:
            empty_seats.append((sheet_name, seat_num, person_name))
    
    return people_without_seats, empty_seats


def main():
    parser = argparse.ArgumentParser(
        description='Check seat assignments between name list and seat graph Excel files'
    )
    parser.add_argument('name_list_file', help='Path to name list Excel file')
    parser.add_argument('seat_graph_file', help='Path to seat graph Excel file')
    parser.add_argument('--name-column', help='Column name for names in name list (auto-detect if not specified)')
    parser.add_argument('--seat-column', help='Column name for seat numbers in seat graph (auto-detect if not specified)')
    parser.add_argument('--seat-name-column', help='Column name for names in seat graph (auto-detect if not specified)')
    parser.add_argument('--output', help='Output file path for results (CSV format)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.name_list_file).exists():
        print(f"Error: Name list file not found: {args.name_list_file}")
        sys.exit(1)
    
    if not Path(args.seat_graph_file).exists():
        print(f"Error: Seat graph file not found: {args.seat_graph_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Excel Seat Checker")
    print("=" * 60)
    
    # Read name list
    print(f"\nReading name list from: {args.name_list_file}")
    name_list = read_name_list(args.name_list_file, args.name_column)
    print(f"Total unique names in name list: {len(name_list)}")
    
    # Read seat graph
    print(f"\nReading seat graph from: {args.seat_graph_file}")
    seat_graph = read_seat_graph(args.seat_graph_file, args.seat_column, args.seat_name_column)
    total_seats = sum(len(seats) for seats in seat_graph.values())
    print(f"Total seats across all sheets: {total_seats}")
    
    # Perform checks
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    people_without_seats, empty_seats = check_seats(name_list, seat_graph)
    
    # Report people without seats
    print(f"\n1. People in name list WITHOUT seats: {len(people_without_seats)}")
    if people_without_seats:
        for name in sorted(people_without_seats):
            print(f"   - {name}")
    else:
        print("   ✓ All people in name list have seats!")
    
    # Report empty seats
    print(f"\n2. Seats WITHOUT corresponding people in name list: {len(empty_seats)}")
    if empty_seats:
        for sheet_name, seat_num, person_name in sorted(empty_seats):
            person_display = person_name if person_name else "(empty)"
            print(f"   - Sheet '{sheet_name}', Seat '{seat_num}': {person_display}")
    else:
        print("   ✓ All seats have corresponding people in name list!")
    
    # Summary statistics
    names_with_seats = name_list - set(people_without_seats)
    print(f"\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total names in list: {len(name_list)}")
    print(f"Names with seats: {len(names_with_seats)}")
    print(f"Names without seats: {len(people_without_seats)}")
    print(f"Total seats: {total_seats}")
    print(f"Empty/unassigned seats: {len(empty_seats)}")
    
    # Save to output file if specified
    if args.output:
        output_data = []
        
        # Add people without seats
        for name in sorted(people_without_seats):
            output_data.append({
                'Type': 'Person Without Seat',
                'Sheet': '',
                'Seat': '',
                'Name': name
            })
        
        # Add empty seats
        for sheet_name, seat_num, person_name in sorted(empty_seats):
            output_data.append({
                'Type': 'Empty Seat',
                'Sheet': sheet_name,
                'Seat': seat_num,
                'Name': person_name if person_name else '(empty)'
            })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
