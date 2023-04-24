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

# Ordered
def pair_with_targetsum(arr, target_sum):
  left, right = 0, len(arr) - 1
  while(left < right):
    current_sum = arr[left] + arr[right]
    if current_sum == target_sum:
      return [left, right]
 
    if target_sum > current_sum:
      left += 1  # we need a pair with a bigger sum
    else:
      right -= 1  # we need a pair with a smaller sum
  return [-1, -1]
 
 
def main():
  print(pair_with_targetsum([1, 2, 3, 4, 6], 6))
  print(pair_with_targetsum([2, 5, 9, 11], 11))
 
 
main()

#Unordered
def pair_with_targetsum(arr, target_sum):
  nums = {}  # to store numbers and their indices
  for i, num in enumerate(arr):
    if target_sum - num in nums:
      return [nums[target_sum - num], i]
    else:
      nums[arr[i]] = i
  return [-1, -1]
 
 
def main():
  print(pair_with_targetsum([1, 2, 3, 4, 6], 6))
  print(pair_with_targetsum([2, 5, 9, 11], 11))
 
 
main()
 