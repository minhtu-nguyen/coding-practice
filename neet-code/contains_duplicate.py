class Solution:
  def containsDuplicate(self, nums: List[int]) -> bool:
    hashset = set()

    for n in nums:
      if n in hashset:
        return True
      hashset.add(n)
    return False

"""
Brute force: 2 loops - O(n^2) and O(1)
Sort first: duplicate adjacent - O(nlogn) [n but sort take nlogn] and O(1)
Extra mem: hashset check and hash - O(n) and O(n)
"""