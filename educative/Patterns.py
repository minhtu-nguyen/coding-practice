### *** Two Pointer ***
'''
As the name suggests, the two pointers pattern uses two pointers to iterate over an array or list until the conditions of the problem are satisfied. This is useful because it allows us to keep track of the values of two different indexes in a single iteration. Whenever there’s a requirement to find two data elements in an array that satisfy a certain condition, the two pointers pattern should be the first strategy to come to mind.
'''
##Valid Palindrome
'''
Write a function that takes a string s as input and checks whether it’s a palindrome or not.
Flow: 
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

##Sum of Three Values
'''
Given an array of integers, nums, and an integer value, target, determine if there are any three integers in nums whose sum is equal to the target, that is, nums[i] + nums[j] + nums[k] == target. Return TRUE if three such integers exist in the array. Otherwise, return FALSE.
Flow: 
- Sort the input array in ascending order.
- Iterate over the entire sorted array to find the triplet whose sum is equal to the target.
- In each iteration, make a triplet by storing the current array element and the other two elements using two pointers (low and high), and calculate their sum.
-Adjust the calculated sum value, until it becomes equal to the target value, by conditionally moving the pointers, low and high.
- Return TRUE if the required sum is found. Otherwise, return FALSE.
Naive approach: The naive approach to solving this problem is to use three nested loops. Each nested loop starts at the index greater than its parent loop. O(n^3) - O(1).
Optimized approach: First, sort the array in ascending order. To find a triplet whose sum is equal to the target value, loop through the entire array. In each iteration:
Store the current array element and set up two pointers (low and high) to find the other two elements that complete the required triplet.
The low pointer is set to the current loop’s index + 1.
The high is set to the last index of the array.
Calculate the sum of array elements pointed to by the current loop’s index and the low and high pointers.
If the sum is equal to target, return TRUE.
If the sum is less than target, move the low pointer forward.
If the sum is greater than target, move the high pointer backward.
Repeat until the loop has processed the entire array. If, after processing the entire array, we don’t find any triplet that matches our requirement, we return FALSE.
Sort = O(nlog(n)) | nested loop = O(n^2) = O(nlog(n) + n^2) = O(n^2) || built-in sort space: O(n)
  nums_lists = [[3, 7, 1, 2, 8, 4, 5],
                [-1, 2, 1, -4, 5, -3],
                [2, 3, 4, 1, 7, 9],
                [1, -1, 0],
                [2, 4, 2, 7, 6, 3, 1]]

  targets = [10, 7, 20, -1, 8]
'''
def find_sum_of_three(nums, target):
    # Sort the input list
    nums.sort()
    # Fix one integer at a time and find the other two
    for i in range(0, len(nums)-2): #distinct: i can’t be equal to low or high, and low can’t be equal to high
        # Initialize the two pointers
        low = i + 1
        high = len(nums) - 1
        # Traverse the list to find the triplet whose sum equals the target
        while low < high:
            triplet = nums[i] + nums[low] + nums[high]
            # The sum of the triplet equals the target
            if triplet == target:
                return True                
            # The sum of the triplet is less than target, so move the low pointer forward
            elif triplet < target:
                low += 1            
            # The sum of the triplet is greater than target, so move the high pointer backward
            else:
                high -= 1    
    # No such triplet found whose sum equals the given target
    return False
##Reverse Words in a String
'''
Given a sentence, reverse the order of its words without affecting the order of letters within a given word.
Flow: 
- Reverse the entire string.
- Start iterating over the reversed string using two pointers, start and end, initially set at 0.
- While iterating over the string, reverse a word pointed by start and end pointers, when a space is found.
- Once the word has been reversed, update the start and end pointer to the index of the space + 1.
- Repeat the process until the entire string is iterated and return the string.
O(n+n)=O(n) | O(n) - copy it into a list of characters to overcome the issue of strings being immutable in Python
    string_to_reverse = ["Hello Friend", "    We love Python",
                         "The quick brown fox jumped over the lazy dog   ",
                         "Hey", "To be or not to be",
                         "AAAAA","Hello     World"]
'''
import re
def reverse_words(sentence):
   # remove leading, trailing and multiple spaces
   sentence = re.sub(' +',' ',sentence.strip())
   # We need to convert the updated string
   # to lists of characters as strings are immutable in Python
   sentence = list(sentence)
   str_len = len(sentence)
   
   #  We will first reverse the entire string.
   str_rev(sentence, 0, str_len - 1)
   #  Now all the words are in the desired location, but
   #  in reverse order: "Hello World" -> "dlroW olleH".
 
   start = 0
   end = 0

   # Now, let's iterate the reversed string and reverse each word in place.
   # "dlroW olleH" -> "World Hello"

   while (start < str_len):
      # Find the end index of the word. 
    while end < str_len and sentence[end] != ' ':
        end += 1
     # Let's call our helper function to reverse the word in-place.
    str_rev(sentence, start, end - 1)
    start = end + 1;
    end += 1
  
   return ''.join(sentence)

# A function that reverses a whole sentence character by character
def str_rev(_str, start_rev, end_rev):
   # Starting from the two ends of the list, and moving
   # in towards the centre of the string, swap the characters
   while start_rev < end_rev:
       temp = _str[start_rev]          # temp store for swapping
       _str[start_rev] = _str[end_rev]  # swap step 1
       _str[end_rev] = temp            # swap step 2


       start_rev += 1                  # Move forwards towards the middle
       end_rev -= 1                    # Move backwards towards the middle