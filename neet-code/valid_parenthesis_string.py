class Solution:
    def isValid(self, s: str) -> bool:
      Map = {")" : "(", "]" : "[", "}" : "{"}
      stack = []

      for c in s:
        if c not in Map:
          stack.append(c)
          continue
        if not stack or stack[-1] != Map[c]:
          return False
        stack.pop()

      return not stack

"""
Use a hashmap 
O(n) go through every element | O(n) use a stack
"""