class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)
        prefix = 1
        for i, n in enumerate(nums):
            res[i] = prefix
            prefix *= n
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res

"""
Prefix and suffix = 2 pass for and back: O(n) and O(n)
Better: store in result array (not count in space) O(n) an O(1)
"""