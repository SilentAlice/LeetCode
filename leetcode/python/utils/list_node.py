"""
Definition for singly-linked list node.
Commonly used in LeetCode problems.
"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"
    
    @classmethod
    def from_list(cls, values):
        """Create a linked list from a list of values."""
        if not values:
            return None
        head = cls(values[0])
        current = head
        for val in values[1:]:
            current.next = cls(val)
            current = current.next
        return head
    
    def to_list(self):
        """Convert linked list to a Python list."""
        result = []
        current = self
        while current:
            result.append(current.val)
            current = current.next
        return result
