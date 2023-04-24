class Solution:
    def missingNumber(self, nums: List[int]) -> int:
      res = len(nums)

      for i in range(len(nums)):
        res += i - nums[i]
      return res
    
'''
a ^ 0 = a | a ^ b ^ a = b (a, a cancel out)
O(1) space --> sum(range) - sum(num)
'''