class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not self.alphanum(s[l]): #Call inside must use self.
                l += 1
            while l < r and not self.alphanum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

    # Could write own alpha-numeric function
    def alphanum(self, c):
        return (
            ord("A") <= ord(c) <= ord("Z")
            or ord("a") <= ord(c) <= ord("z")
            or ord("0") <= ord(c) <= ord("9")
        )
"""
Built-in extra space: O(n)
def isPalindrome(s):
  newStr = ""

  for c in s:
    if c.isalnum():
      nenewStr += c.lower()
  return nenewStr == nenewStr[::-1]
Better no extra space: use 2 pointers
"""
# Grokking
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    print("\tThe element being pointed by the left index is", s[left], sep = " ")
    print("\tThe element being pointed by the right index is", s[right], sep = " ")
    while left < right:
        print("\tWe check if the two elements are indeed the same, in this case...")
        if s[left] != s[right]:  # If the elements at index l and index r are not equal,
            print("\tThe elements aren't the same, hence we return False")
            return False    # then the symmetry is broken, the string is not a palindrome
        print("\tThey are the same, thus we move the two pointers toward the middle to continue the \n\tverification process.\n")
        left = left + 1  # Heading towards the right
        right = right - 1  # Heading towards the left
        print("\tThe new element at the left pointer is", s[left], sep = " ")
        print("\tThe new element at the right pointer is", s[right], sep = " ")
    # We reached the middle of the string without finding a mismatch, so it is a palindrome.
    return True


# Driver Code
def main():

    test_cases = ["RACEACAR", "A", "ABCDEFGFEDCBA",
                  "ABC", "ABCBA", "ABBA", "RACEACAR"]
    for i in range(len(test_cases)):
        print("Test Case #", i + 1)
        print("-" * 100)
        print("\tThe input string is '", test_cases[i], "' and the length of the string is ", len(test_cases[i]), ".", sep='')
        print("\nIs it a palindrome?.....", is_palindrome(test_cases[i]))
        print("-" * 100)


if __name__ == '__main__':
    main()

































