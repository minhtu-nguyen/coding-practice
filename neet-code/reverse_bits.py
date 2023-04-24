class Solution:
    def reverseBits(self, n: int) -> int:
      res = 0
      for i in range(32):
        bit = (n >> i) & 1
        res += (bit << (31 - 1))
      return res

"""
Get the last bit and place it at the right place using shift
AND OR <<
"""