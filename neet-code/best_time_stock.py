class Solution:
    def maxProfit(self, prices: List[int]) -> int:
      res = 0

      l = 0
      for r in range(1, len(prices)):
        if prices[r] < prices[l]:
          l=r
        res = max(res, prices[r] - prices[l])
      return res

"""
2 pointer: O(1)
Old: 
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
      l, r = 0, 1
      maxP = 0

      while r < len(prices):
        if prices[l] < prices[r]:
          profit = prices[r] - prices[l]
          maxP = max(maxP, profit)
        else:
          l = r
        r += 1
      return maxP
"""