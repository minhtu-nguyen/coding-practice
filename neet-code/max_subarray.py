class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub = nums[0]
        curSum = 0

        for n in nums:
            if curSum < 0:
                curSum = 0
            curSum += n
            maxSub = max(maxSub, curSum)
        return maxSub

"""
O(n^3): 2 loop traverse, 1 loop to sum
O(n^2): curSum in 2nd loop
O(n): check prefix < 0, reset curSum
"""