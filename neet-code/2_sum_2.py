class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
      l, r = 0, len(numbers) - 1

      while l < r:
        curSum = numbers[l] + numbers[r]
        if curSum > target:
          r = -1
        elif curSum < target:
          l += 1
        else:
          return [l + 1, r + 1]

"""
Brute force: O(n) - try all combinations (stop if > target)
Better: eliminating elems from the end of array --> 2 pointers
"""