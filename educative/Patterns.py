### *** Two Pointer ***
##Valid Palindrome
'''
Write a function that takes a string s as input and checks whether it’s a palindrome or not.
Understanding: 
- Initialize two pointers and traverse them towards the middle of the string.
- Check if the current pair of elements is identical or not.
- If they are not identical, return False, else, move both pointers by one index toward the middle.
- If we reach the middle of the string without finding a mismatch, return True.
Naive approach: reverse the string and then compare the reversed string with the original string. This, despite being a solution of linear time complexity, will still occupy extra space, and we can use an optimized approach to save extra space.  
Optimized approach: We initialize two pointers and move them from opposite ends. The first pointer starts moving toward the middle from the start of our string, while the second pointer starts moving toward the middle from the end of our string. This allows us to compare these elements at every single instance to find a non-matching pair, and if they reach the middle of the string without encountering any non-matching pair, that means we indeed traverse a palindrome.
O(n)
O(1)
test_cases = ["RACEACAR", "A", "ABCDEFGFEDCBA",
                  "ABC", "ABCBA", "ABBA", "RACEACAR"]
'''
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left = left + 1 
        right = right - 1
    return True