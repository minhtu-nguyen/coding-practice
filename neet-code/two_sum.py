class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
      prevMap = {} # val -> index

      for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
          return [prevMap[diff], i]
        prevMap[n] = i

"""
Brute force: 2 loops - O(n^2) and O(1)
Better: one solution sum - n = target --> HashMap
Less efficient HM: initialize first then look up
Better: Visit then add | check if exist the target - O(n) and O(n)
"""