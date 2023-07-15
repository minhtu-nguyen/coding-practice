def twoSum(nums, target):
    prevMap = {}

    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[n], i]
        prevMap[n] = i

def containsDuplicate(nums):
    hashset = set()

    for n in nums:
      if n in hashset:
          return True
      hashset.add(n)
    return False

def isAnagram(s, t):
    if len(s) != len(t):
        return False
    
    counterS, counterT = {}, {}

    for i in range(len(s)):
        counterS[s[i]] = 1 + counterS.get(s[i], 0)
        counterT[s[i]] = 1 + counterT.get(s[i], 0)
    return counterS == counterT

def isPalindrome(s):
    l, r = 0, len(s) - 1

    while l < r:
        while l < r and not isalphanum(s[l]):
            l += 1
        while l < r and not isalphanum(a[r]):
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1
        r -= 1
    return True

def isalphanum(c):
    return (
        ord("a") <= ord(c) <= ord("z")
        or ord("A") <= ord(c) <= ord("Z")
        or ord("0") <= ord(c) <= ord("9")
    )

def stock(prices):
    res = 0

    lowest = prices[0]
    for price in prices:
        if price < lowest:
            lowest = price
        res = max(res, price - lowest)
    return res

def validParen(s):
    map = {")": "(", "]": "[", "}": "{"}
    stack = []

    for c in s:
        if c not in map:
            stack.append(c)
        if not stack or stack[-1] != map[c]:
            return False
        stack.pop()
    return not stack

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def reverseList(head):
    prev, curr = None, head

    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev

def mergeTwoLists(list1, list2):
    