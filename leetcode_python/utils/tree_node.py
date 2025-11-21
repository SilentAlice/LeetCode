"""
Definition for binary tree node.
Commonly used in LeetCode problems.
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"
    
    @classmethod
    def from_list(cls, values):
        """Create a binary tree from a list representation (level-order)."""
        if not values or values[0] is None:
            return None
        
        root = cls(values[0])
        queue = [root]
        i = 1
        
        while queue and i < len(values):
            node = queue.pop(0)
            
            if i < len(values) and values[i] is not None:
                node.left = cls(values[i])
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] is not None:
                node.right = cls(values[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    def to_list(self):
        """Convert binary tree to level-order list representation."""
        if not self:
            return []
        
        result = []
        queue = [self]
        
        while queue:
            node = queue.pop(0)
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)
        
        # Remove trailing None values
        while result and result[-1] is None:
            result.pop()
        
        return result
