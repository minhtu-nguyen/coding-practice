class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            n &= n - 1 #res += n % 2            
            res += 1    # n = n >> 1
        return res

"""
%2 then >> O(1)
n = n & (n - 1)
"""