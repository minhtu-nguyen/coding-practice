class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
      if len(s) != len(t):
        return False

      countS, countT = {}, {}

      for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0) # Not exist --> error | use default value 0 to fix
        countT[t[i]] = 1 + countT.get(t[i], 0)

      return countS == countT # the same number of characters even duplicate --> same number of keys in dict

"""
Brute force: 
HashMap: O(S + T) and O(S + T)
Built-in: return Counter(s) == Counter(t)
Not use extra space: return sorted(s) == sorted(t)
"""