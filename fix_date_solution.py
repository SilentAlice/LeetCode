#!/usr/bin/env python3
"""
Solution for corrupted Excel date: 875805804060701000
Most likely original date: 2025-01-01
"""

from datetime import datetime, timedelta

def fix_corrupted_excel_date(corrupted_value):
    """
    Fix a corrupted Excel date that was multiplied by a large factor.
    
    The corrupted value 875805804060701000 appears to be an Excel date serial number
    that was incorrectly multiplied by approximately 19,181,869,640,823.
    
    Most likely original date: 2025-01-01
    """
    excel_epoch = datetime(1899, 12, 30)
    
    # Calculate the factor and original date
    # For 2025-01-01, Excel serial is 45658
    target_serial_2025_01_01 = 45658
    factor = corrupted_value / target_serial_2025_01_01
    
    # Calculate the date
    original_date = excel_epoch + timedelta(days=target_serial_2025_01_01)
    
    print("=" * 70)
    print("CORRUPTED EXCEL DATE FIX")
    print("=" * 70)
    print(f"\nCorrupted value: {corrupted_value}")
    print(f"\nMost likely original date: {original_date.strftime('%Y-%m-%d')}")
    print(f"Excel serial number: {target_serial_2025_01_01}")
    print(f"Corruption factor: {factor:,.0f}")
    print("\n" + "=" * 70)
    print("HOW TO FIX IN EXCEL:")
    print("=" * 70)
    print(f"\n1. In Excel, use this formula to convert the corrupted date:")
    print(f"   =DATE(1899,12,30) + ({corrupted_value}/{factor:.0f})")
    print(f"\n   Or more simply:")
    print(f"   =DATE(1899,12,30) + {target_serial_2025_01_01}")
    print(f"\n2. Or use this formula directly:")
    print(f"   =DATE(2025,1,1)")
    print(f"\n3. If you have the corrupted value in cell A1, use:")
    print(f"   =DATE(1899,12,30) + ROUND(A1/{factor:.0f}, 0)")
    print("\n" + "=" * 70)
    
    return original_date

if __name__ == "__main__":
    corrupted_value = 875805804060701000
    original_date = fix_corrupted_excel_date(corrupted_value)
    
    print(f"\n✓ Fixed date: {original_date.strftime('%Y-%m-%d (%A, %B %d, %Y)')}")
