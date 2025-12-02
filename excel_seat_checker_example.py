#!/usr/bin/env python3
"""
Example usage of excel_seat_checker.py

This script demonstrates how to use the Excel seat checker programmatically.
"""

from excel_seat_checker import read_name_list, read_seat_graph, check_seats, normalize_name

# Example: Using the functions directly in your code
def example_usage():
    # File paths
    name_list_file = "name_list.xlsx"
    seat_graph_file = "seat_graph.xlsx"
    
    # Read the files
    print("Reading name list...")
    name_list = read_name_list(name_list_file, name_column="Name")
    
    print("Reading seat graph...")
    seat_graph = read_seat_graph(
        seat_graph_file,
        seat_column="Seat Number",
        name_column="Person Name"
    )
    
    # Perform checks
    people_without_seats, empty_seats = check_seats(name_list, seat_graph)
    
    # Process results
    print(f"\nFound {len(people_without_seats)} people without seats")
    print(f"Found {len(empty_seats)} empty seats")
    
    # You can further process these results as needed
    return people_without_seats, empty_seats


if __name__ == '__main__':
    example_usage()
