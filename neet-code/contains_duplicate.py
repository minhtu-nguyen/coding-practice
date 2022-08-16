class Solution:
  def containsDuplicaet(self, nums: List[int]) -> bool:
    hashset = set()

    for n in nums:
      if n in hashset:
        return True
      hashset.add(n)
    return False

"""
Brute force: 2 loops - O(n^2) and O(1)
Sort first: O(nlogn) and O(1)
Extra mem: hashset - O(n) and O(n)
"""