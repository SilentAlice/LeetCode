#!/usr/bin/env python3
"""
Utility to fix corrupted Excel dates.
The number 875805804060701000 appears to be a corrupted date from 2025.
This script tries various conversion methods to recover the original date.
"""

from datetime import datetime, timedelta
import sys

def excel_serial_to_date(serial):
    """Convert Excel serial number to date (Excel epoch: 1900-01-01)"""
    # Excel incorrectly treats 1900 as a leap year, so we adjust
    excel_epoch = datetime(1899, 12, 30)
    return excel_epoch + timedelta(days=int(serial))

def unix_timestamp_to_date(timestamp, unit='seconds'):
    """Convert Unix timestamp to date"""
    if unit == 'seconds':
        return datetime.fromtimestamp(timestamp)
    elif unit == 'milliseconds':
        return datetime.fromtimestamp(timestamp / 1000)
    elif unit == 'microseconds':
        return datetime.fromtimestamp(timestamp / 1000000)
    elif unit == 'nanoseconds':
        return datetime.fromtimestamp(timestamp / 1000000000)
    else:
        return None

def analyze_corrupted_date(corrupted_value):
    """Try various interpretations of the corrupted date"""
    print(f"Analyzing corrupted date value: {corrupted_value}")
    print("=" * 70)
    
    results = []
    
    # Debug: Show the number in different formats
    print(f"\nNumber analysis:")
    print(f"  As string: {str(int(corrupted_value))}")
    print(f"  Length: {len(str(int(corrupted_value)))} digits")
    print(f"  Scientific notation: {corrupted_value:.2e}")
    
    # Check if "2025" appears in the string representation
    corrupted_str = str(int(corrupted_value))
    if "2025" in corrupted_str:
        pos = corrupted_str.find("2025")
        print(f"  Found '2025' at position {pos}")
        # Try to extract date around this position
        for offset in [-4, -2, 0, 2, 4]:
            try:
                start = max(0, pos + offset)
                if start + 8 <= len(corrupted_str):
                    date_str = corrupted_str[start:start+8]
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                        date = datetime(year, month, day)
                        results.append((f"Found '2025' in string (offset {offset})", 
                                      date.strftime("%Y-%m-%d %H:%M:%S")))
            except:
                pass
    
    # Method 1: Direct Excel serial number (unlikely but worth trying)
    try:
        date = excel_serial_to_date(corrupted_value)
        results.append(("Excel Serial (direct)", date.strftime("%Y-%m-%d %H:%M:%S")))
    except:
        pass
    
    # Method 2: Divide by common factors (in case it was multiplied)
    # Try a wider range of factors
    factors_to_try = [10**i for i in range(3, 18)]  # 1000 to 10^17
    for factor in factors_to_try:
        try:
            divided = corrupted_value / factor
            if 1 <= divided <= 100000:  # Reasonable Excel serial range
                date = excel_serial_to_date(divided)
                if 2020 <= date.year <= 2030:
                    results.append((f"Excel Serial (÷{factor})", date.strftime("%Y-%m-%d %H:%M:%S")))
        except:
            pass
    
    # Method 3: Unix timestamp in various units
    for unit in ['seconds', 'milliseconds', 'microseconds', 'nanoseconds']:
        try:
            date = unix_timestamp_to_date(corrupted_value, unit)
            if 2020 <= date.year <= 2030:  # Reasonable year range
                results.append((f"Unix Timestamp ({unit})", date.strftime("%Y-%m-%d %H:%M:%S")))
        except (ValueError, OSError):
            pass
    
    # Method 4: Try dividing Unix timestamp interpretations
    for factor in [1000, 1000000, 1000000000]:
        for unit in ['seconds', 'milliseconds']:
            try:
                divided = corrupted_value / factor
                date = unix_timestamp_to_date(divided, unit)
                if 2020 <= date.year <= 2030:
                    results.append((f"Unix Timestamp (÷{factor}, {unit})", 
                                  date.strftime("%Y-%m-%d %H:%M:%S")))
            except (ValueError, OSError):
                pass
    
    # Method 5: Check if it's days in milliseconds (common Excel corruption)
    days_in_ms = corrupted_value / 86400000
    if 1 <= days_in_ms <= 100000:
        try:
            excel_epoch = datetime(1899, 12, 30)
            date = excel_epoch + timedelta(days=days_in_ms)
            if 2020 <= date.year <= 2030:
                results.append(("Days in Milliseconds (Excel)", 
                              date.strftime("%Y-%m-%d %H:%M:%S")))
        except:
            pass
    
    # Method 6: Check if it's a date that was concatenated or formatted incorrectly
    # Sometimes dates get corrupted as strings that are then converted to numbers
    corrupted_str = str(int(corrupted_value))
    if len(corrupted_str) >= 8:
        # Try interpreting as YYYYMMDD
        for start_pos in range(len(corrupted_str) - 7):
            try:
                date_str = corrupted_str[start_pos:start_pos+8]
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                    date = datetime(year, month, day)
                    results.append((f"String interpretation (pos {start_pos})", 
                                  date.strftime("%Y-%m-%d %H:%M:%S")))
            except:
                pass
    
    # Method 7: Remove trailing zeros (common in Excel corruption)
    corrupted_str = str(int(corrupted_value))
    # Try removing trailing zeros
    for num_zeros in range(1, min(10, len(corrupted_str))):
        try:
            trimmed = corrupted_str.rstrip('0')
            if len(trimmed) >= 8:
                # Try interpreting trimmed version
                for start_pos in range(len(trimmed) - 7):
                    try:
                        date_str = trimmed[start_pos:start_pos+8]
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                            date = datetime(year, month, day)
                            results.append((f"Trimmed zeros (pos {start_pos})", 
                                          date.strftime("%Y-%m-%d %H:%M:%S")))
                    except:
                        pass
        except:
            pass
    
    # Method 8: Try if it's Excel date + time fraction multiplied
    # Excel stores dates as days, times as fractions (0.0 to 0.999...)
    # Sometimes this gets corrupted
    for factor in [10**i for i in range(10, 20)]:
        try:
            divided = corrupted_value / factor
            if 45000 <= divided <= 50000:  # Range for 2023-2037 dates
                date = excel_serial_to_date(divided)
                if 2020 <= date.year <= 2030:
                    results.append((f"Excel Serial Extended (÷{factor})", 
                                  date.strftime("%Y-%m-%d %H:%M:%S")))
        except:
            pass
    
    # Display results
    if results:
        print("\nPossible original dates:")
        print("-" * 70)
        for i, (method, date_str) in enumerate(results, 1):
            print(f"{i}. {method:40s} -> {date_str}")
        
        # Highlight 2025 dates
        print("\n" + "=" * 70)
        print("Dates in 2025 (most likely candidates):")
        print("-" * 70)
        for method, date_str in results:
            if "2025" in date_str:
                print(f"  {method:40s} -> {date_str}")
    else:
        print("\nNo reasonable date interpretations found.")
        print("The value might need manual investigation or additional context.")
    
    return results

if __name__ == "__main__":
    corrupted_value = 875805804060701000
    
    if len(sys.argv) > 1:
        try:
            corrupted_value = float(sys.argv[1])
        except ValueError:
            print(f"Error: Could not parse '{sys.argv[1]}' as a number")
            sys.exit(1)
    
    analyze_corrupted_date(corrupted_value)
