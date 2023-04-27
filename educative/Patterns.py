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

### *** Fast and Slow
'''
Unlike the two pointers approach, which is concerned with data values, the fast and slow pointers approach is used to determine data structure traits using indices in arrays or node pointers in linked lists. The approach is commonly used to detect cycles in the given data structure, so it’s also known as Floyd’s cycle detection algorithm.
'''
##Happy Number
'''
Write an algorithm to determine if a number n is happy.

- A happy number is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy.
- Return TRUE if n is a happy number, and FALSE if not.
Flow: 
- Initialise a variable slow with the input number and fast with the squared sum of the input number’s digits.
- If fast is not 1 and also not equal to slow, increment slow by one iteration and fast by two iterations. Essentially, set slow to Sum of Digits(slow) and fast to Sum of Digits(Sum of Digits(fast)).
- If fast converges to 1, return TRUE, otherwise return FALSE.
Naive approach: The brute force approach is to repeatedly calculate the squared sum of digits of the input number and store the computed sum in a hash set. For every calculation, we check if the sum is already present in the set. If yes, we've detected a cycle and should return FALSE. Otherwise, we add it to our hash set and continue further. If our sum converges to, we've found a happy number. O(logn) | O(n)
Optimized approach: We maintain track of two values using a slow pointer and a fast pointer. The slow runner advances one number at each step, while the fast runner advances two numbers. We detect if there is any cycle by comparing the two values and checking if the fast runner has indeed reached the number one. We return True or False depending on if those conditions are met. 2 pointers = O(logn+logn) = O(logn) | O(1)
    inputs = [1, 5, 19, 25, 7]
'''
def is_happy_number(n):

    # Helper function that calculates the sum of squared digits.
    def sum_of_squared_digits(number):
        total_sum = 0
        while number > 0:
            number, digit = divmod(number, 10)
            total_sum += digit ** 2
        return total_sum

    slow_pointer = n 
    fast_pointer = sum_of_squared_digits(n)  

    while fast_pointer != 1 and slow_pointer != fast_pointer: 
        slow_pointer = sum_of_squared_digits(slow_pointer)
        fast_pointer = sum_of_squared_digits(sum_of_squared_digits(fast_pointer))

    if(fast_pointer == 1):
        return True
    return False

## Linked List Cycle
'''
Check whether or not a linked list contains a cycle. If a cycle exists, return TRUE. Otherwise, return FALSE. The cycle means that at least one node can be reached again by traversing the next pointer.
Flow: 
- Initialize both fast and slow pointers to the head of the linked list.
- Move the slow pointer one node ahead and the fast pointer two nodes ahead.
- Check if both pointers point to the same node at any point. If yes, then return TRUE.
- Else, if the fast pointer reaches the end of the linked list, return FALSE.
Naive approach: The naive approach is to traverse the linked list and store the current node in a set. At each iteration, we check if the current node is already present in the set. If it is, we’ve found a cycle and return TRUE. Otherwise, we add the node to the set. If the traversal has been completed, return FALSE, since we’ve reached the end of the linked list without encountering a cycle. O(n) - O(n)
Optimized approach: To recap, the solution to this problem can be divided into the following three parts:
- Initialize both the slow and fast pointers to the head node.
- Move both pointers at different rates, i.e. the slow pointer will move one step ahead whereas the fast pointer will move two steps.
- If both pointers are equal at some point, we know that a cycle exists.
O(n) - O(1)
    input = (
             [2, 4, 6, 8, 10, 12],
             [1, 3, 5, 7, 9, 11],
             [0, 1, 2, 3, 4, 6],
             [3, 4, 7, 9, 11, 17],
             [5, 1, 4, 9, 2, 3],
            )
    pos = [0, -1, 1, -1, 2]
'''
def detect_cycle(head):
    if head is None:
        return False

    # Initialize two pointers, slow and fast, to the head of the linked list
    slow, fast = head, head
    
    # Run the loop until we reach the end of the
    # linked list or find a cycle
    while fast and fast.next:
        # Move the slow pointer one step at a time
        slow = slow.next
        # Move the fast pointer two steps at a time
        fast = fast.next.next
        
        # If there is a cycle, the slow and fast pointers will meet
        if slow == fast:
            return True
    # If we reach the end of the linked list and haven't found a cycle, return False          
    return False

## Middle of the Linked List
'''
Given a singly linked list, return the middle node of the linked list. If the number of nodes in the linked list is even, return the second middle node.
Flow: 
- Initialize two pointers slow and fast at the head of the linked list.
- Traverse the linked list, moving the slow pointer one step forward and fast pointer two steps forward.
- When the fast pointer reaches the last node or NULL, then the slow pointer will point to the middle node of the linked list.
- Return the slow pointer.
Naive approach: The naive approach is to count the number of nodes in the linked list first, and then find the middle node in the next iteration.
Optimized approach: 
- Initialize two pointers named slow and fast at the head of the linked list.
- While traversing, move the slow pointer one step forward and the fast pointer two steps forward.
- When the fast pointer reaches the last node or NULL, then the slow pointer will point to the middle node of the linked list.
O(n) - O(1)
    input = (
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [3, 2, 1],
        [10],
        [1, 2],
    )
'''
def get_middle_node(head):
    # Initially slow and fast pointers point to head
    slow = head
    fast = head
    # Traverse the linked list until fast reaches at the last node or NULL
    while fast and fast.next:
        # Move slow pointer one step ahead
        slow = slow.next
        # Move fast pointer two step ahead
        fast = fast.next.next
    # Return the slow pointer
    return slow

## Circular Array Loop
'''
An input array, nums containing non-zero integers, is given, where the value at each index represents the number of places to skip forward (if the value is positive) or backward (if the value is negative). When skipping forward or backward, wrap around if you reach either end of the array. For this reason, we are calling it a circular array. Determine if this circular array has a cycle. A cycle is a sequence of indices in the circular array characterized by the following:

The same set of indices is repeated when the sequence is traversed in accordance with the aforementioned rules.
The length of the sequence is at least two.
The loop must be in a single direction, forward or backward.
Flow: 
- Traverse the entire nums array using slow and fast pointers, starting from index 0.
- Move the slow pointer one time forward/backward and the fast pointer two times forward/backward.
- If loop direction changes at any point, continue to the next element.
- If the direction does not change, check whether both pointers meet at the same node, if yes, then the loop is detected and return TRUE.
- Return FALSE if don’t encounter a loop after traversing the whole array.
Naive approach: The naive approach is to traverse the whole array and check for each element whether we can form a cycle starting from each element or not. We’ll run a loop on every array element and keep track of the visited element using an additional array. We’ll check the condition for both the forward and backward cycles. If the direction of the cycle changes at any point, we’ll come out of that loop and continue verifying the loop condition for the remaining elements. O(n^2) - O(n)
Optimized approach: 
- Move the slow pointer x steps forward/backward, where x is the value at the ith index of the array.
- Move the fast pointer x steps forward/backward and do this again for the value at the (i + 1)th index of the array.
- Return TRUE when both pointers meet at the same point.
- If the direction changes at any point or taking a step returns to the same location, then follow the steps above for the next element of the array.
- Return FALSE if we have traversed every element of the array without finding a loop.
O(n) - O(1)
    input = (
            [-2, -3, -9],
            [-5, -4, -3, -2, -1],
            [-1, -2, -3, -4, -5],
            [2, 1, -1, -2],
            [-1, -2, -3, -4, -5, 6],
            [1, 2, -3, 3, 4, 7, 1]
            )
'''
def circular_array_loop(nums):
  # Set slow and fast pointer at first element.
  slow = fast = 0
  size = len(nums)
  for i in range(1, size + 1):
    # Save slow pointer's value before moving.
    prev = slow
    # Move slow pointer to one step.
    slow = next_step(slow, nums[slow], size)
    # Check if cycle is impossible, then set both pointers to next value
    # and move to the next iteration.
    if is_not_cycle(nums, prev, slow):
      fast, slow = i, i
      continue

    # This flag indicates whether we need to move to the next iteration.
    next_iter = False
    # Number of moves of fast pointer
    moves = 2
    for _ in range(moves):
      # Save fast pointer's value before moving.
      prev = fast
      # Move fast pointer check cycle after every move.
      fast = next_step(fast, nums[fast], size)
      # If cycle is not possible, set slow and fast to next element
      # set 'next_iter' to true and break this loop.
      if is_not_cycle(nums, prev, fast):
        fast, slow = i, i
        next_iter = True
        break
    
    # Move to the next iteration
    if next_iter:
      continue
    
    # If both pointers are at same position after moving both, a cycle is detected.
    if slow == fast:
      return True
      
  return False

# A function to calculate the next step
def next_step(pointer, value, size):
    result = (pointer + value) % size
    if result < 0:
        result += size
    return result

# A function to detect a cycle doesn't exist
def is_not_cycle(nums, prev, pointer):
    # If signs of both pointers are different or moving a pointer takes back to the same value,
    # then the cycle is not possible, we return True, otherwise return False.
    if (nums[prev] >= 0 and nums[pointer] < 0) or (abs(nums[pointer] % len(nums)) == 0):
        return True
    else:
        return False

##Palindrome Linked List
'''
For the given head of the linked list, find out if the linked list is a palindrome or not. Return TRUE if the linked list is a palindrome. Otherwise, return FALSE.
Flow: 
- Initialize both the slow and fast pointers as head.
- Traverse linked list using both pointers at different speeds. At each iteration, the slow pointer increments by one node, and the fast pointer increments by two nodes.
- Continue doing so until the fast pointer reaches the end of the linked list. At this instance, the slow pointer will be pointing at the middle of the linked list.
- Reverse the second half of the linked list and compare it with the first half.
- If both halves are the same then the given linked list is a palindrome.
O(n) - O(1)
    input = (
                [2, 4, 6, 4, 2],
                [0, 3, 5, 5, 0],
                [9, 7, 4, 4, 7, 9],
                [5, 4, 7, 9, 4, 5],
                [5, 9, 8, 3, 8, 9, 5],
            )
'''
# Check palindrome in linked_list
def palindrome(head):
    # Initialize variables with head
    slow = head
    fast = head
    revert_data = None
    mid_node = head

    # Traverse linked_list through fast and slow
    # pointers to get the middle node
    while fast and fast.next:
        mid_node = slow
        slow = slow.next
        fast = fast.next.next

    # Fast pointer of odd linked list will point to last node
    # of linked list but it will point to NULL for even linked list
    saved_odd_mid_node = None
    if fast:
        saved_odd_mid_node = slow
        slow = slow.next

    # It will skip the first half
    mid_node.next = None
    # Pass middle node as a head to reverse function
    # to revert the next half of linked_list
    revert_data = reverse_linked_list(slow)
    # Pass second half reverted data to compare_two_halves
    # function to check the palindrome property
    check = False
    check = compare_two_halves(head, revert_data)
    # Revert second half back to the original linked list
    revert_data = reverse_linked_list(revert_data)

    # Connect both halves
    # If linked list was of odd sized, connect the middle node
    # first which was skipped while reverting the second half
    if saved_odd_mid_node:
        mid_node.next = saved_odd_mid_node
        saved_odd_mid_node.next = revert_data
    else:
        mid_node.next = revert_data

    # Return true if there's only one node
    # or both are pointing to NULL
    if head is None or revert_data is None:
        return True
    if check:
        return True
    return False


def compare_two_halves(first_half, second_half):
    while first_half is not None and second_half is not None:
        if first_half.data != second_half.data:
            return False
        else:
            first_half = first_half.next
            second_half = second_half.next
    return True 

# Template to reverse the linked list

def reverse_linked_list(slow_ptr):
    reverse = None
    while slow_ptr is not None:
        next = slow_ptr.next
        slow_ptr.next = reverse
        reverse = slow_ptr
        slow_ptr = next
    return reverse

### *** Sliding Window
'''
A window is a sublist formed over a part of an iterable data structure. It can be used to slide over the data in chunks corresponding to the window size. The sliding window pattern allows us to process the data in segments instead of the entire list. The segment or window size can be set according to the problem’s requirements.
'''
## Repeated DNA Sequences
'''
Given a string, s, that represents a DNA sequence, and a number, k, return all the contiguous sequences (substrings) of length k that occur more than once in the string. The order of the returned subsequences does not matter. If no repeated subsequence is found, the function should return an empty set.
Flow: 
- Iterate over the input string.
- Create a window of length k  to extract a substring.
- Add the computed content of the window (the hash or the substring itself) to the set that keeps track of all the unique substrings of length k.
- Move the window one step forward and add the computed content of the new window to a set.
- If the computed content of a window is already present in the set, the substring is repeated. Append it to an output list.
- Continue the iteration over the string until all possible substrings have been processed, and return the output list.
Naive approach: A naive approach would be to iterate through the input DNA sequence and add all the unique sequences of length k into a set. If a sequence is already in a set, it is a repeated sequence. To achieve this, we can use two nested loops. The outer loop will iterate over each position in the DNA sequence, and the inner loop will create all possible substrings of length k starting at that position. O((n−k+1)×k).
Optimized approach: We use the ***Rabin-Karp algorithm, which utilizes a sliding window with ***rolling hash for pattern matching. Instead of recomputing the hash for each window slide, we can use the rolling hash technique, where we can simply subtract the hash value of the character being removed from the window and add the hash value of the character being added. There are multiple approaches for computing hash values, and the choice of the hash function can impact the algorithm’s time complexity.
- Hashing and comparison in linear time: sums the ASCII code of characters present
- Hashing and comparison in constant time: Polynomial rolling hash technique

- Iterate the string.
- Compute the hash value for the contents of the window.
- Add this hash value to the set that keeps track of the hashes of all substrings of the given length.
- Move the window one step forward and compute a new hash value.
- If the hash value of a window has already been seen, the sequence in this window is repeated, so we add it to our output set.
- Once all characters of the string have been traversed, we return the output set.
O(n−k+1) - O(n−k+1)

    inputs_string = ["ACGT", "AGACCTAGAC", "AAAAACCCCCAAAAACCCCCC", "GGGGGGGGGGGGGGGGGGGGGGGGG",
                     "TTTTTCCCCCCCTTTTTTCCCCCCCTTTTTTT", "TTTTTGGGTTTTCCA",
                     "AAAAAACCCCCCCAAAAAAAACCCCCCCTG", "ATATATATATATATAT"]
    inputs_k = [3, 3, 8, 10, 13, 30, 30, 21]
'''
def find_repeated_sequences(s, k):
    window_size = k
    if len(s) <= window_size:
        return set()
    base = 4
    hi_place_value = pow(base, window_size - 1)
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    numbers = []
    for i in range(len(s)):
        numbers.append(mapping.get(s[i]))
    hashing = 0
    substring_hashes, output = set(), set()
    for start in range(len(s) - window_size + 1):
        if start != 0:
            hashing = (hashing - numbers[start - 1] * hi_place_value) * base \
                + numbers[start + window_size - 1]
        else:
            for end in range(window_size):
                hashing = hashing * base + numbers[end]
        if hashing in substring_hashes:
            output.add(s[start:start + window_size])
        substring_hashes.add(hashing)
    return output

## Find Maximum in Sliding Window
'''
Given an integer list, nums, find the maximum values in all the contiguous subarrays (windows) of size w.
Flow:
- Create a collection (of your own choice) to process the elements of the input list.
- Process the first window such that, at the end of the iteration, the elements of the first window are present in your collection in descending order.
- When the first window has been processed, add the first element from your collection to the output list, since this will be the maximum in the first window.
- Process the remaining windows using the same logic used to process the first window.
- In each iteration, add the first element from your collection to the output list and slide the window one step forward.
- Return the output list.
Naive approach: The naive approach is to slide the window over the input list and find the maximum in each window separately. O(n(2w+wlogw))=O(n(wlogw)) - O(w)
Optimized approach: 
- First, we validate the inputs. If the input list is empty, we return an empty list and if the window size is greater than the list length, we set the window to be the same size as the input list.
- Then, we process the first w elements of the input list. We will use a deque to store the indexes of the candidate maximums of each window.
- For each element, we perform the clean-up step, removing the indexes of the elements from the deque whose values are smaller than or equal to the value of the element we are about to add to the deque. Then, we append the index of the new element to the back of the deque.
- After the first w elements have been processed, we append the element whose index is present at the front of the deque to the output list as it is the maximum in the first window.
- After finding the maximum in the first window, we iterate over the remaining input list. For each element, we repeat Step 3 as we did for the first w elements.
- Additionally, in each iteration, before appending the index of the current element to the deque, we check if the first index in the deque has fallen out of the current window. If so, we remove it from the deque.
- Finally, we return the list containing the maximum elements of each window.
O(n) - O(w)
    window_sizes = [3, 3, 3, 3, 2, 4, 3, 2, 3, 18]
    nums_list = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [1, 5, 8, 10, 10, 10, 12, 14, 15, 19, 19, 19, 17, 14, 13, 12, 12, 12, 14, 18, 22, 26, 26, 26, 28, 29, 30],
        [10, 6, 9, -3, 23, -1, 34, 56, 67, -1, -4, -8, -2, 9, 10, 34, 67],
        [4, 5, 6, 1, 2, 3],
        [9, 5, 3, 1, 6, 3],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [-1, -1, -2, -4, -6, -7],
        [4, 4, 4, 4, 4, 4]
    ]
'''
from collections import deque

# function to clean up the deque
def clean_up(i, current_window, nums):
    while current_window and nums[i] >= nums[current_window[-1]]:
        current_window.pop()

# function to find the maximum in all possible windows
def find_max_sliding_window(nums, w):
    if len(nums) == 0:
        return []
    output = []
    current_window = deque()
    if w > len(nums):
        w = len(nums)
    for i in range(w):
        clean_up(i, current_window, nums)
        current_window.append(i)
    output.append(nums[current_window[0]])
    for i in range(w, len(nums)):
        clean_up(i, current_window, nums)
        if current_window and current_window[0] <= (i - w):
            current_window.popleft()
        current_window.append(i)
        output.append(nums[current_window[0]])
    return output

## Minimum Window Subsequence
'''
Given strings str1 and str2, find the minimum (contiguous) substring sub_str of str1, such that every character of str2 appears in sub_str in the same order as it is present in str2. If there is no window in str1 that covers all characters in str2, return an empty string. If there are multiple minimum-length windows, return the one with the leftmost starting index.
Flow:
- Initialize two pointers index_s1 and index_s2 to 0 to iterate both the strings.
- If the character pointed by index_s1 in s1 is the same as the character pointed by index_s2 in s2, increment both pointers.
- Create two new pointers (start and end) once index_s2 reaches the end of s2.
- Set the start to the value of index_s1 and end to start + 1.
- Decrement the start pointer until index_s2 becomes less than 0.
- Decrement index_s2 only if the character pointed by the start pointer in s1 is equal to the character pointed to by index_s2 in s2.
- Calculate the length of a substring by subtracting values of the end and start variables.
- If this length is less than the current minimum length, update the length variable and min_subsequence string.
- Repeat until index_s1 reaches the end of s1.
Naive approach: The naive approach would be considering all possible windows of str1 and checking which windows contain str2. Out of all the windows in str1, we’ll choose the window with the shortest length. O(n^3) - O(1)
Optimized approach: 
- Initialize two pointers, index_s1 and index_s2, to zero for iterating both strings.
- If the character pointed by index_s1 in str1 is the same as the character pointed by index_s2 in str2, increment both pointers. Otherwise, only increment index_s1.
- Create two new pointers (start and end) once index_s2 reaches the end of str2. With these two pointers, we’ll slide the window in the opposite direction.
- Set the start to the value of index_s1 and end to start + 1.
- Decrement index_s2 if the character pointed by the start pointer in str1 is equal to the character pointed to by index_s2 in str2 and decrement the start pointer until index_s2 becomes less than zero.
- Calculate the length of a substring by subtracting values of the end and start variables.
- If this length is less than the current minimum length, update the length variable and min_subsequence string.
- Repeat until index_s1 reaches the end of str1.
O(m×n) - O(1)
'''
def min_window(str1, str2):
    size_str1, size_str2 = len(str1), len(str2)
    length = float('inf')
    index_s1, index_s2 = 0, 0
    min_subsequence = ""
    while index_s1 < size_str1:
        if str1[index_s1] == str2[index_s2]:
            index_s2 += 1
            if index_s2 == size_str2:
                start, end = index_s1, index_s1+1
                index_s2 -= 1
                while index_s2 >= 0:
                    if str1[start] == str2[index_s2]:
                        index_s2 -= 1
                    start -= 1
                start += 1
                if end - start < length:
                    length = end - start
                    min_subsequence = str1[start:end]
                index_s1 = start
                index_s2 = 0
        index_s1 += 1
    return min_subsequence
## Minimum Window Substring
'''
We are given two strings, s and t. The minimum window substring of t in s is defined as follows:
It is the shortest substring of s that includes all of the characters present in t.
The frequency of each character in this substring that belongs to t should be equal to or greater than its frequency in t.
The order of the characters does not matter here.
We have to find the minimum window substring of t in s.
Flow: 
- Set up a sliding, adjustable window to move across the string s.
- Initialize two collections: one to store the frequency of characters in t and the other to track the frequency of characters in the current window.
- Iterate over s, expanding the current window until the frequencies of characters of t in the window are at least equal to their respective frequencies in t.
- Trim the window by removing all the unnecessary characters. If the current window size is less than the length of the minimum window substring found so far, update the minimum window substring.
- Continue iterating over s and perform the previous two steps in each iteration.
- Return the minimum window substring.
Naive approach: The naive approach would be to find all possible substrings of s and then identify the shortest substring that contains all characters of t with corresponding frequencies equal to or greater than those in t. O(n^2) - O(n)
Optimized approach: 
-We validate the inputs. If t is an empty string, we return an empty string.
-Next, we initialize two hash maps: req_count, to save the frequency of characters in t, and window, to keep track of the frequency of characters of t in the current window. We also initialize a variable, required, to hold the number of unique characters in t. Lastly, we initialize current which keeps track of the characters that occur in t whose frequency in the current window is equal to or greater than their corresponding frequency in t.
-Then, we iterate over s and in each iteration we perform the following steps:
If the current character occurs in t, we update its frequency in the window hash map.
If the frequency of the new character is equal to its frequency in req_count, we increment current.
If current is equal to required, we decrease the size of the window from the start. As long as current and required are equal, we decrease the window size one character at a time, while also updating the minimum window substring. Once current falls below required, we slide the right edge of the window forward and move on to the next iteration.
- Finally, when s has been traversed completely, we return the minimum window substring.
O(m+(N*m)) - O(m)
    s = ["PATTERN", "LIFE", "ABRACADABRA", "STRIKER", "DFFDFDFVD"]
    t = ["TN", "I", "ABC", "RK", "VDD"]
'''
def min_window(s, t):
    # empty string scenario
    if t == "":
        return ""
    
    # creating the two hash maps
    req_count = {}
    window = {}

    # populating req_count hash map
    for c in t:
        req_count[c] = 1 + req_count.get(c, 0)

    # populating window hash map
    for c in t:
        window[c] = 0

    # setting up the conditional variables
    current, required = 0, len(req_count)
    
    # setting up a variable containing the result's starting and ending point
    # with default values and a length variable
    res, res_len = [-1, -1], float("infinity")
    
    # setting up the sliding window pointers
    left = 0
    for right in range(len(s)):
        c = s[right]

        # if the current character also occurs in t, update its frequency in window hash map
        if c in t:
            window[c] = 1 + window.get(c, 0)

        # updating the current variable
        if c in req_count and window[c] == req_count[c]:
            current += 1

        # adjusting the sliding window
        while current == required:
            # update our result
            if (right - left + 1) < res_len:
                res = [left, right]
                res_len = (right - left + 1)
            
            # pop from the left of our window
            if s[left] in t:
                window[s[left]] -= 1

            # if the popped character was among the required characters and
            # removing it has reduced its frequency below its frequency in t, decrement current
            if s[left] in req_count and window[s[left]] < req_count[s[left]]:
                current -= 1
            left += 1
    left, right = res

    # return the minimum window substring
    return s[left:right+1] if res_len != float("infinity") else ""

## Longest Substring without Repeating Characters
'''
Given a string, str, return the length of the longest substring without repeating characters.
Flow:
- Initialize an empty hash map and a variable to track character indexes and the start of the window, respectively.
- Traverse the string character by character. For each character, if the hash map does not contain the current character, store it with its index as the value.
- If the hash map contains the current character and its index falls within the current window, a repeating character is found. Otherwise, store it in the hash map with its index as the value.
- When a repeating character is found, update the start of window to the previous location of the current element and increment it. Also, calculate the length of the current window.
- Update the longest substring seen so far if the length of the current window is greater than its current value.
- Return the length of the longest substring.
Naive approach: For each substring, we check whether any character in it is repeated. After checking all the possible substrings, the substring with the longest length that satisfies the specified condition is returned. O(n^3) - O(min(m,n))
Optimized approach: 
-Traverse the input string.
-Use a hash map to store elements along with their respective indexes.
-If the current element is present in the hash map, check whether it’s already present in the current window. If it is, we have found the end of the current window and the start of the next. We check if it’s longer than the longest window seen so far and update it accordingly.
-Store the current element in the hash map with the key as the element and the value as the current index.
-At the end of the traversal, we have the length of the longest substring with all distinct characters.
O(n) - O(n)
    str = [
        "abcabcbb",
        "pwwkew",
        "bbbbb",
        "ababababa",
        "",
        "ABCDEFGHI",
        "ABCDEDCBA",
        "AAAABBBBCCCCDDDD",
    ]
'''
def find_longest_substring(str):
    # Check the length of str
    if len(str) == 0:
        return 0

    window_start, longest, window_length = 0, 0, 0

    last_seen_at = {}

    # Traverse str to find the longest substring
    # without repeating characters.
    for index, val in enumerate(str):
        # If the current element is not present in the hash map,
        # then store it in the hash map with the current index as the value.
        if val not in last_seen_at:
            last_seen_at[val] = index
        else:
            # If the current element is present in the hash map,
            # it means that this element may have appeared before.
            # Check if the current element occurs before or after `window_start`.
            if last_seen_at[val] >= window_start:
                window_length = index - window_start
                if longest < window_length:
                    longest = window_length
                window_start = last_seen_at[val] + 1

            # Update the last occurrence of
            # the element in the hash map
            last_seen_at[val] = index

    index += 1
    # Update the longest substring's
    # length and starting index.
    if longest < index - window_start:
        longest = index - window_start

    return longest

## Minimum Size Subarray Sum
'''
Given an array of positive integers nums and a positive integer target, find the window size of the shortest contiguous subarray whose sum is greater than or equal to the target value. If no subarray is found, 0 is returned.
Flow:
- Initialize a variable to store the size of the minimum subarray.
- Iterate the input array.
- In each iteration, add up an element of an array.
- If the sum of the current subarray ≥ the target value, compare the previous window size with the current one and store the smaller value.
- Find a smaller subarray that is greater than or equal to the target value by sliding the window.
- Repeat the process until the entire array is iterated.
O(n) - O(1)
    target = [7, 4, 11, 10, 5, 15]
    input_arr = [[2, 3, 1, 2, 4, 3],
                      [1, 4, 4], [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 3, 4], [1, 2, 1, 3],
                      [5, 4, 9, 8, 11, 3, 7, 12, 15, 44]]
'''
def min_sub_array_len(target, nums):
    # Initializing window_size to a max number
    window_size = float('inf')
    # Initialize start pointer to 0 and sum to 0
    start = 0
    total_sum = 0
    # Iterate over the input array
    for end in range(len(nums)):
        total_sum += nums[end]
        # check if we can remove elements from the start of the subarray
        # while still satisfying the target condition
        while total_sum >= target:
            # Finding size of current subarray
            curr_sub_arr_size = (end + 1) - start
            window_size = min(window_size, curr_sub_arr_size)
            total_sum -= nums[start]
            start += 1

    if window_size != float('inf'):
        return window_size
    else:
        return 0

### *** Merge Intervals
'''
Each interval is represented by a start and an end time. The most common problems solved using this pattern are scheduling problems. The key to understanding this pattern and exploiting its power lies in understanding how any two intervals may overlap.
'''
## Merge Intervals
'''
We are given an array of closed intervals, intervals, where each interval has a start time and an end time. The input array is sorted with respect to the start times of each interval. Your task is to merge the overlapping intervals and return a new output array consisting of only the non-overlapping intervals.
Flow:
- Insert the first interval from the input list into the output list.
- Start a loop to iterate over each interval of the input list, except for the first.
- If the input interval is overlapping with the last interval in the output list, merge these two intervals and replace the last interval of the output list with this merged interval.
- Otherwise, if the input interval does not overlap with the last interval in the output list, add the input interval to the output list.
Naive approach: The naive approach is to start from the first interval in the input list and check for any other interval in the list that overlaps it. If there is an overlap, merge the other interval into the first interval and then remove the other interval from the list. Repeat this for all the remaining intervals in the list. O(n^2) - O(1)
Optimized approach: 
- Insert the first interval from the input list into the output list.
- Traverse the input list of intervals. For each interval in the input list, we do the following:
  * If the input interval is overlapping with the last interval in the output list, merge these two intervals and replace the last interval of the output list with this merged interval.
  * Otherwise, add the input interval to the output list.
O(n) - O(1)
'''
class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.closed = True  # by default, the interval is closed
    # set the flag for closed/open

    def set_closed(self, closed):
        self.closed = closed

    def __str__(self):
        return "[" + str(self.start) + ", " + str(self.end) + "]" \
            if self.closed else \
                "(" + str(self.start) + ", " + str(self.end) + ")"
    
def merge_intervals(intervals):
    if not intervals:
        return None
    result = []
    result.append(Interval(intervals[0].start, intervals[0].end))
    for i in range(1, len(intervals)):
        last_added_interval = result[len(result) - 1]
        cur_start = intervals[i].start
        cur_end = intervals[i].end
        prev_end = last_added_interval.end
        if cur_start <= prev_end:
            last_added_interval.end = max(cur_end, prev_end)
        else:
            result.append(Interval(cur_start, cur_end))
    return result
## Insert Interval
'''
You’re given a list of non-overlapping intervals, and you need to insert another interval into the list. Each interval is a pair of non-negative numbers, the first being the start time and the second being the end time of the interval. The input list of intervals is sorted in ascending order of start time. The intervals in the output must also be sorted by the start time, and none of them should overlap. This may require merging those intervals that now overlap as a result of the addition of the new interval.
Flow: 
- Create a new list to store the output and add the intervals that start before the new interval.
- Add the new interval to the output and merge it with the previously added interval.
- Add the other intervals to the output and merge with the previously added intervals when they overlap with each other.
Naive approach: The naive approach is that we mark each interval in the list as non-visited. Now, for each interval that is marked as non-visited, we search if another interval exists in the list which overlaps in any form with the current interval. If that interval exists, we merge both of these intervals and mark them visited. If it doesn't overlap with anyone, we mark that interval as visited as well. O(n^2)
Optimized approach: We create a new list where we store all the existing intervals that occur before the start of the interval to be inserted. When we finally run into an overlapping interval, we merge the two together and append the remaining intervals into our output array. O(n) - O(1)
'''
def insert_interval(existing_intervals, new_interval):
    new_start, new_end = new_interval.start, new_interval.end
    i = 0
    n = len(existing_intervals)
    output = []
    while i < n and existing_intervals[i].start < new_start:
        output.append(existing_intervals[i])
        i = i + 1
    if not output or output[-1].end < new_start:
        output.append(new_interval)
    else:
        output[-1].end = max(output[-1].end, new_end)
    while i < n:
        ei = existing_intervals[i]
        start, end = ei.start, ei.end
        if output[-1].end < start:
            output.append(ei)
        else:
            output[-1].end = max(output[-1].end, end)
        i += 1
    return output

## Interval List Intersections
'''
For two lists of closed intervals given as input, interval_list_a and interval_list_b, where each interval has its own start and end time, write a function that returns the intersection of the two interval lists.
Flow:
- Compare the starting and ending times of a given interval from A and B.
- If the start time of the current interval in A is less than or equal to the end time of the current interval in B, or vice versa, we have found an intersection. Add it to a resultant list.
- Move forward in the list whose current interval ends earlier.
- Repeat comparison and move forward steps to find all intersecting intervals.
- Return the resultant list of intersecting intervals.
Naive approach: use a nested loop for finding intersecting intervals.The outer loop will iterate for every interval in interval_list_a and the inner loop will search for any intersecting interval in the interval_list_b.If such an interval exists, we add it to the intersections list. O(n^2)
Optimized approach: 
- Compare the starting and ending times of a given interval from A and B.
- If the start time of the current interval in A is less than or equal to the end time of the current interval in B, or vice versa, we have found an intersection. Add it to a resultant list.
- Move forward in the list whose current interval ends earlier and repeat comparison and moving forward steps to find all intersecting intervals.
- Return the resultant list of intersecting intervals.
O(n + m) - O(1)
'''
# Function to find the intersecting points between two intervals
def intervals_intersection(interval_list_a, interval_list_b):
    intersections = []  # to store all intersecting intervals
    # index "i" to iterate over the length of list a and index "j"
    # to iterate over the length of list b
    i = j = 0

    # while loop will break whenever either of the lists ends
    while i < len(interval_list_a) and j < len(interval_list_b):
        # Let's check if interval_list_a[i] intersects interval_list_b[j]
        #  1. start - the potential startpoint of the intersection
        #  2. end - the potential endpoint of the intersection
        start = max(interval_list_a[i].start, interval_list_b[j].start)
        end = min(interval_list_a[i].end, interval_list_b[j].end)

        if start <= end:    # if this is an actual intersection
            intersections.append(Interval(start, end))   # add it to the list

        # Move forward in the list whose interval ends earlier
        if interval_list_a[i].end < interval_list_b[j].end:
            i += 1
        else:
            j += 1

    return intersections

## Employee Free Time
'''
You’re given a list containing the schedules of multiple people. Each person’s schedule is a list of non-overlapping intervals in sorted order. An interval is specified with the start time and the end time, both being positive integers. Your task is to find the list of intervals representing the free time for all the people. We’re not interested in the interval from negative infinity to zero or from the end of the last scheduled interval in the input to positive infinity.
Flow: 
- Add all the employee schedules to a list.
- Sort each schedule in the list by starting time.
- Iterate over the intervals, keeping track of the previous latest ending time.
- If the starting time for any period occurs after the previous latest ending time, then this is a free time interval for all employees and we can add it to the result list.
- Repeat the steps to track the latest ending time and compare with the new starting time to collect all common free intervals in the result list.
- Return the result list.
We find all common free time intervals by merging all of the intervals and finding out whether there are any gaps or intervals available in between the timeslots.
- We add all the employee schedules to one list and then sort that list by starting time, in ascending order.
- The intervals are then iterated while keeping track of the previous latest ending time.
- If the starting time for any period is later than the previous latest ending time, we have discovered a free time interval for all employees, which we add to the result.
O(nlogn) - O(n)
'''
def employee_free_time(schedule):
    # Initializing two lists
    ans = []
    intervals = []

    # Merging all the employee schedules into one list of intervals
    for s in schedule:
        intervals.extend(s)

    # Sorting all intervals
    intervals.sort(key=lambda x: x.start)
    # Initializing prev_end as the endpoint of the first interval
    prev_end = intervals[0].end
    # iterating through the intervals and adding the gaps we find to the answer list
    for interval in intervals:
        if interval.start > prev_end:
            ans.append(Interval(prev_end, interval.start))
        # if the current interval's ending time is later than the current prev_end, update it
        prev_end = max(prev_end, interval.end)
    return ans

### *** In-place Reversal of a Linked List
'''
We iterate in a linked list and keep track of the current node, the next node, and the previous node simultaneously. Keeping track of the nodes allows us to easily change the links between them and make them point to a different node than before.
'''
## Reverse Linked List
'''
Given the head of a singly linked list, reverse the linked list and return its updated head.
Flow:
- Traverse the linked list.
- Append the last node before the head node.
- Point head to the new added node.
- Repeat until the linked list is not NULL.
Naive approach: The naive approach to this problem is to clone the linked list into another data structure like an array or push the nodes of the list to the stack, and then create a new linked list. After pushing all the nodes to the stack, starting from the first node, the stack will return the nodes in reverse order when we pop all the nodes. We can use the popped node to create the new linked list with reversed nodes. O(n) - O(n)
Optimized approach: 
- Initialize two pointers. The first pointer points to the head of the linked list, and the second to the node following the head node.
-Traverse through the linked list. At each step, we are essentially dealing with a pair of consecutive linked list nodes, and we are just doing two things:
  a. Reverse the order of the current pair of nodes by changing the next pointer of the second node to point to the first node.

  b. Move to the next pair of nodes by updating both pointers.
- Continue the traversal until the first pointer reaches the last node in the linked list.
O(n) - O(1)
'''
# Template for linked list node class

class LinkedListNode:
    # __init__ will be used to make a LinkedListNode type object.
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

# Template for the linked list
class LinkedList:
    # __init__ will be used to make a LinkedList type object.
    def __init__(self):
        self.head = None

    # insert_node_at_head method will insert a LinkedListNode at head
    # of a linked list.
    def insert_node_at_head(self, node):
        if self.head:
            node.next = self.head
            self.head = node
        else:
            self.head = node

    # create_linked_list method will create the linked list using the
    # given integer array with the help of InsertAthead method.
    def create_linked_list(self, lst):
        for x in reversed(lst):
            new_node = LinkedListNode(x)
            self.insert_node_at_head(new_node)
    
    # __str__(self) method will display the elements of linked list.
    def __str__(self):
        result = ""
        temp = self.head
        while temp:
            result += str(temp.data)
            temp = temp.next
            if temp:
                result += ", "
        result += ""
        return result
    
def reverse(head):
    if not head or not head.next:
        return head
        
    list_to_do = head.next
    reversed_list = head
    reversed_list.next = None
    
    while list_to_do:
        temp = list_to_do
        list_to_do = list_to_do.next
        temp.next = reversed_list
        reversed_list = temp

    return reversed_list

## Reverse Nodes in k-Group
'''
Given a linked list, reverse the nodes of the linked list k at a time and return the modified list. Here, k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k, the nodes left in the end will remain in their original order.
- Count and check if there are k number of nodes in a linked list.
- Reverse the set of k nodes.
- Reconnect the reversed set of k nodes with the rest of the linked list.
- Repeat the process till less than k or no nodes are left in the linked list.
Naive approach: A naive approach would be to use another data structure—like a stack—to reverse the nodes of the linked list and then create a new linked list with reversed nodes. O(n) - O(n + k)
Optimized approach: 
- Count and check if there are k number of nodes in a linked list.
- Reverse the set of k nodes.
- Reconnect the reversed set of k nodes with the rest of the linked list.
- Repeat the process till less than k or no nodes are left in the linked list.
O(n) - O(1)
'''
# reverse will reverse the k number of nodes in the linked list
def reverse(head, k):
    previous, current, next = None, head, None
    index = 0
    while current and index < k:
        next = current.next
        current.next = previous
        previous = current
        current = next
        index += 1
    return previous, current, next


# find_length will find the total length of the linked list
def find_length(start):
    current = start
    count = 0
    while current:
        current = current.next
        count += 1
    return count

# reverse_linked_list is our challenge function that will reverse
# the group of k nodes in the linked list
def reverse_linked_list(head, k):
    if k <= 1 or not head:
        return head
    i, count = 0, 0
    current, previous = head, None
    total_length = find_length(head)
    while True:
        i += 1
        last_node_of_previous_part = previous
        last_node_of_current_part = current
        next = None  
        previous, current, next = reverse(last_node_of_current_part, k)
        count += k
        if last_node_of_previous_part:
            last_node_of_previous_part.next = previous
        else:
            head = previous
        last_node_of_current_part.next = current

        if current is None or (total_length - count) < k:
            break
        previous = last_node_of_current_part
    return head

## Reverse Linked List II
'''
You’re given the head of a singly linked list with n nodes and two positive integers, left and right. Our task is to reverse the list’s nodes from position left to position right and return the reversed list.
Flow: 
- Find the left position node.
- Reverse the linked list from left to right position node.
- Merge the reversed list with the rest of the linked list.
Naive approach: The naive approach to this problem is to clone the given linked list into another data structure, like an array or stack, and then create a new linked list. When we do this, the group of nodes from left to right will be reversed, and the new head of the reversed group of nodes will be stored in an array. In the end, we can complete the linked list using the new head by pointing the previous node from the original linked list to the new head and pointing the tail of the reversed group to the remaining original linked list. O(n) - O(n)
Optimized approach: above. O(n) - O(1)
'''
# Assume that the linked list has left to right nodes.
# Reverse left to right nodes of the given linked list.
def reverse(head, left, right):
    rev_head = None
    ptr = head  # a pointer to traverse the original list.
    while right >= left:
        # Track the next node to traverse in the original list
        next = ptr.next

        # At the beginning of the reversed list,
        # insert the node pointed to by `ptr`
        ptr.next = rev_head
        rev_head = ptr

        # Move on to the next node
        ptr = next

        # Decrement the count of nodes to be reversed by 1
        right -= 1
    # Return reversed list's head
    return rev_head
def reverse_between(head, left, right):
    ptr = head  # a pointer to traverse the original list.
    # a pointer to keep the track of previous node
    previous = None
    reverse_head = None
    right_node = None
    # Keep traversing until left and right node number
    count = 1
    # Move the ptr to the left number node
    while count < left and ptr:
        previous = ptr  # keep track of the previous node
        ptr = ptr.next
        count += 1
    if ptr:
        # keep track of the next node outside the [left - right] 
        # interval
        next_node = ptr
        while count <= right and next_node:
            # keep track of the right number node
            right_node = next_node  
            next_node = next_node.next
            count += 1
        # If we have found the left till right nodes, then we 
        # reverse them.
        if right_node:
            # Reverse these [left-right] nodes and get the new head
            #  of the reversed list
            reverse_head = reverse(ptr, left, right)
        if previous:
            # point previous.next to the reversed linked list
            previous.next = reverse_head
        if next_node:
            # traverse in the reversed linked list until the last node
            tmp = reverse_head
            while tmp.next:
                tmp = tmp.next
            # add the next node to the reversed linked list
            tmp.next = next_node

    # We will reverse head if there are node before the [left-right]
    # position interval
    if previous:
        return head
    # We will simply return the reverse head if there is no node
    # before the [left-right] position interval
    else:
        return reverse_head

## Reorder List
'''
Given the head of a singly linked list, reorder the list as if it were folded on itself.
Flow:
- Find the middle node.
- If there are two middle nodes, choose the second node.
- Reverse the second half of the linked list.
- Merge the first and second half of the linked list.
Naive approach: First, we need to start from the first node using the current pointer. To find the last node, we traverse the complete linked list and add the last node in front of the current node. After adding the last node, the current node will move to the next node. Each time the last node is added, the current node will move ahead in the list. For this reason we need to find the last node. We’ll end the program when both current and last nodes become equal. O(n^2) - O(1)
Optimized approach: 
- Find the middle node. If there are two middle nodes, then choose the second node.
- Reverse the second half of the linked list.
- Merge both halves of the linked lists alternatively.
O(n) - O(1)
'''
def reorder_list(head):
    if not head:
        return head
    
    # find the middle of linked list
    # in 1->2->3->4->5->6 find 4 
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next 
        
    # reverse the second part of the list
    # convert 1->2->3->4->5->6 into 1->2->3 and 6->5->4
    # reverse the second half in-place
    prev, curr = None, slow
    while curr:
        curr.next, prev, curr = prev, curr, curr.next       

    # merge two sorted linked lists
    # merge 1->2->3 and 6->5->4 into 1->6->2->5->3->4
    first, second = head, prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next
    
    return head

## Swapping Nodes in a Linked List
'''
Given the linked list and an integer k, return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end of the linked list. We’ll number the nodes of the linked list starting from 1 to n.
Flow: 
- Traverse the linked list.
- Find the kth node from the start of the linked list.
- Count the total number of nodes in the linked list.
- Find the position of the kth node from the end by subtracting k from the total number of nodes.
- Move to the kth node from the end.
- Swap the values of both nodes.
Three-pass approach | Two-pass approach
One pass approach
- Traverse the linked list using the current pointer and count the length of the linked list simultaneously.
- Find the first kth node.
- After finding the first kth node, use another pointer, end, to traverse in a linked list.
- Move the end pointer until the current pointer doesn’t reach the end of the linked list
- The end pointer now points to the end node.
O(n) - O(1)
'''
# Template for swapping two nodes of the linked list

def swap(node1, node2):
    temp = node1.data
    node1.data = node2.data
    node2.data = temp
# swap_nodes is used to implement the complete solution
def swap_nodes(head, k):
    # count will be used to count the number of nodes in a linked list
    count = 0
    # front will be used to point the kth node at the beginning of the
    # linked list end will be used to point the kth node at the end of
    # the linked list
    front, end = None, None
    # curr will be used to traverse in a linked list
    curr = head
    # Traverse in a linked list until the null node
    while curr:
        # Increment the length for each node
        count += 1
        # If end is not null it means the kth node from the beginning has
        # found now, we can move the end pointer in the linked list to
        # find the kth node from the end of the linked list
        if end is not None:
            end = end.next
        # if the count has become equal to k, it means the curr is
        # pointing the kth node at the beginning of the linked list
        if count == k:
            # Set front to the curr node
            front = curr
            # Set end to the head node
            end = head
        # move curr to the next node
        curr = curr.next
    # swap the values of two nodes: front and end
    swap(front, end)

    return head

## Reverse Nodes In Even Length Groups
'''
You’re given a linked list. Your task is to reverse all of the nodes that are present in the groups with an even number of nodes in them. The nodes in the linked list are sequentially assigned to non-empty groups whose lengths form the sequence of the natural numbers (1,2,3,4...). The length of a group is the number of nodes assigned to it. In other words:
The 1st node is assigned to the first group.
The 2nd and 3rd nodes are assigned to the second group.
The 4th, 5th, and 6th nodes are assigned to the third group and so on.
Flow:
- Start traversing the second node of the list if it exists, since the first node does not need to be reversed as it is part of an odd numbered group of 1 node.
- Keep traversing the linked list till l nodes have been traversed, where l is the length of the current group of nodes (which starts from 2).
- During this traversal, keep a count n of the nodes being traversed. This is to ensure that if the linked list ends before l nodes have been traversed, we still have a track of the n nodes that have been traversed.
- If the value of n is odd, do not reverse this group. Otherwise, if the value of n is even, reverse the n nodes in this group.
- Increment l by 1 and repeat this process until the linked list has been traversed completely.
Approach:
We need to reverse each group that has an even number of nodes in it. We can think of each of these groups of nodes as separate linked lists. So, for each of these groups applying an in-place linked list reversal solves the original problem. We need to invoke the in-place reversal of linked list code for every group with an even number of nodes present. However, we’ll need to modify the terminating condition of the loop that is used in the in-place reversal of a linked list pattern. We constantly need to check if the current group even requires any sort of reversal or not and if it is the last group do we even need to modify it or not.
'''
def reverse_even_length_groups(head):
    prev = head  # Node immediately before the current group
    l = 2  # The head doesn't need to be reversed since it's a group of one node, so starts with length 2
    while prev.next:
        node = prev
        n = 0
        for i in range(l):
            if not node.next:
                break
            n += 1
            node = node.next
        if n % 2:  # odd length
            prev = node
        else:      # even length
            reverse = node.next
            curr = prev.next
            for j in range(n):
                curr_next = curr.next
                curr.next = reverse
                reverse = curr
                curr = curr_next
            prev_next = prev.next
            prev.next = node
            prev = prev_next
        l += 1
    return head

### *** Two Heaps
'''
In some problems, we’re given a set of data such that it can be divided into two parts. We can either use the first part to find the smallest element using the min-heap and the second part to find the largest element using the max-heap or vice versa. There might be cases where we need to find the two largest numbers from two different data sets. We’ll use two max-heaps to store two different data sets in that case.
'''
## Maximize Capital
'''
You need to develop a program for making automatic investment decisions for a busy investor. The investor has some start-up capital, c, to invest and a portfolio of projects in which they would like to invest in. The investor wants to maximize their cumulative capital as a result of this investment.
To help them with their decision, they have information on the capital requirement for each project and the profit it’s expected to yield. For example, if project A has a capital requirement of 3, and the investor’s current capital is 1 then the investor can invest in this project. Now, supposing that the project yields a profit of 2, the investor’s capital at the end of the project will be 1+2=3. The investor can now choose to invest in project A as well since their current capital has increased.
As a basic risk-mitigation measure, the investor would like to set a limit on the number of projects, k, they invest in. For example, if the value of k is 2, then we need to identify the two projects that the investor can afford to invest in, given their capital requirements, and that yield the maximum profits.
Further, these are one-time investment opportunities, that is, the investor can only invest once in a given project.
Flow:
- Create a min-heap for capitals.
- Identify the projects that can be invested in that fall within the range of the current capital.
- Select the project that yields the highest profit.
- Add the profit earned to the current capital.
- Repeat until k projects have been selected.
Naive approach: traverse every value of the capitals array based on the available capital. If the current capital is less than or equal to the capital value in the array, then store the profit value in a new array that corresponds to the capital index. Whenever the current capital becomes less than the capital value in the array, we’ll call the max function to get the largest profit value. The selected profit value will be added to the previous capital. Repeat this process until we get the required number of projects containing maximum profit. O(n^2) - O(n)
Optimized approach: 
- Push all required capital values onto a min-heap.
- Select projects in the range of the current capital and push their profits onto a max-heap.
- Select the project from the max-heap that yields the maximum profit.
- Add the profit from the selected project to the current capital.
- Repeat steps 2–4 until we have selected the required number of projects.
O(nlogn + nlogn) = O(nlogn) - O(n)
'''
from heapq import heappush, heappop


def maximum_capital(c, k, capitals, profits):
    current_capital = c
    capitals_min_heap = []
    profits_max_heap = []

    for x in range(0, len(capitals)):
        heappush(capitals_min_heap, (capitals[x], x))

    for _ in range(k):

        while capitals_min_heap and capitals_min_heap[0][0] <= current_capital:
            c, i = heappop(capitals_min_heap)
            heappush(profits_max_heap, (-profits[i], i))
        
        if not profits_max_heap:
            break

        j = -heappop(profits_max_heap)[0]
        current_capital = current_capital + j

    return current_capital

## Find Median from a Data Stream
'''
Implement a data structure that’ll store a dynamically growing list of integers and provide access to their median in O(1).
Flow: 
- Split up incoming numbers into two lists—small and large. Those that are smaller than the current middle element are small, and those that are larger than it are large.
- Maintain these lists as heaps so that the root of the small heap is the largest element in it and the root of large heap is the smallest element in it.
- If the size of the large heap is greater than the size of small heap or, if size of small heap is greater than the size of the large heap + 1, rebalance the heaps.
- If the number of elements is even, the median is the mean of the root of the two heaps. Else, it’s the root of the small heap.
Naive approach: The naive solution is to first sort the data and then find the median. Insertion sort is an algorithm that can be used to sort the data as it appears. O(n^2) - O(1)
Optimized approach: 
- Use a min-heap to store the larger 50% of the numbers seen so far and a max-heap for the smaller 50% of the numbers.
- Add the incoming elements to the appropriate heaps.
- Calculate the median using the top elements of the two heaps.
O(logn) - O(1)
'''
class MedianOfStream:

    max_heap_for_smallnum = []
    min_heap_for_largenum = []

    def insert_num(self, num):
        if not self.max_heap_for_smallnum or -self.max_heap_for_smallnum[0] >= num:
            heappush(self.max_heap_for_smallnum, -num)
        else:
            heappush(self.min_heap_for_largenum, num)

        if len(self.max_heap_for_smallnum) > len(self.min_heap_for_largenum) + 1:
            heappush(self.min_heap_for_largenum, -heappop(self.max_heap_for_smallnum))
        elif len(self.max_heap_for_smallnum) < len(self.min_heap_for_largenum):
            heappush(self.max_heap_for_smallnum, -heappop(self.min_heap_for_largenum))

    def find_median(self):
        if len(self.max_heap_for_smallnum) == len(self.min_heap_for_largenum):

            # we have even number of elements, take the average of middle two elements
            # we divide both numbers by 2.0 to ensure we add two floating point numbers
            return -self.max_heap_for_smallnum[0] / 2.0 + self.min_heap_for_largenum[0] / 2.0

        # because max-heap will have one more element than the min-heap
        return -self.max_heap_for_smallnum[0] / 1.0

## Sliding Window Median
'''
Given an integer array, nums, and an integer, k, there is a sliding window of size k, which is moving from the very left to the very right of the array. We can only see the k numbers in the window. Each time the sliding window moves right by one position. Given this scenario, return the median of the each window. 
Flow:
- Declare a min-heap and a max-heap to store the elements of the sliding window.
- Push k elements onto the max-heap and transfer 2/k numbers (the higher numbers) to the min-heap.
- Compute the median of the window elements. If the window size is even, it’s the mean of the top of the two heaps. If it’s odd, it’s the top of the max-heap.
- Move the window forward and rebalance the heaps.
- If the incoming number is less than the top of the max-heap, push it on to the max-heap, else, push it onto the min-heap.
- If the outgoing number is at the top of either of the heaps, remove it from that heap.
- Repeat the steps to calculate the median, add the incoming number, rebalance the heaps, and remove the outgoing number from the heaps.
Naive approach: use nested loops to traverse over the array. The outer loop ranges over the entire array, and the nested loop is used to iterate over windows of k elements. For each window, we’ll first sort the elements and then compute the median. We’ll append this value to the median list and move the window one step forward. Using quicksort O((n - k)logk) - O((n - k)logk).
Optimized approach: 
- Populate max-heap with k elements.
- Transfer k/2elements from the max-heap to the min-heap.
- If the window size is odd, the median is the top of the max-heap. Otherwise, it’s the mean of the top elements of the two heaps.
- Move the window forward and add the outgoing number in the hash map, which is used to track the outgoing numbers.
- Rebalance the heaps if they have more elements.
- If the top element of the max-heap or the min-heap is present in the hash map with a frequency greater than 0, this element is irrelevant. We remove it from the respective heap and the hash map.
- Repeat the process until all elements are processed.
O(nlogn) - O(n)
'''
def median_sliding_window(nums, k):
    # To store the medians
    medians = []

    # To keep track of the numbers that need to be removed from the heaps
    outgoing_num = {}

    # Max heap
    small_list = []

    # Min heap
    large_list = []

    # Initialize the max heap by multiplying each element by -1
    for i in range(0, k):
        heappush(small_list, -1 * nums[i])

    # Transfer the top 50% of the numbers from max heap to min heap
    # while restoring the sign of each number
    for i in range(0, k//2):
        element = heappop(small_list)
        heappush(large_list, -1 * element)

    i = k
    while True:
        # If the window size is odd
        if (k & 1) == 1:
            medians.append(float(small_list[0] * -1))
        else:
            medians.append((float(small_list[0] * -1) + float(large_list[0])) * 0.5)

        # Break the loop if all elements have been processed
        if i >= len(nums):
            break

        # Outgoing number
        out_num = nums[i - k]

        # Incoming number
        in_num = nums[i]
        i += 1

        # Variable to keep the heaps balanced
        balance = 0

        # If the outgoing number is from max heap
        if out_num <= (small_list[0] * -1):
            balance -= 1
        else:
            balance += 1

        # Add/Update the outgoing number in the hash map
        if out_num in outgoing_num:
            outgoing_num[out_num] = outgoing_num[out_num] + 1
        else:
            outgoing_num[out_num] = 1

        # If the incoming number is less than the top of the max heap, add it in that heap
        # Otherwise, add it in the min heap
        if small_list and in_num <= (small_list[0] * -1):
            balance += 1
            heappush(small_list, in_num * -1)
        else:
            balance -= 1
            heappush(large_list, in_num)

        # Re-balance the heaps
        if balance < 0:
            heappush(small_list, (-1 * large_list[0]))
            heappop(large_list)
        elif balance > 0:
            heappush(large_list, (-1 * small_list[0]))
            heappop(small_list)

        # Remove invalid numbers present in the hash map from top of max heap
        while (small_list[0] * -1) in outgoing_num and (outgoing_num[(small_list[0] * -1)] > 0):
            outgoing_num[small_list[0] * -1] = outgoing_num[small_list[0] * -1] - 1
            heappop(small_list)

        # Remove invalid numbers present in the hash map from top of min heap
        while large_list and large_list[0] in outgoing_num and (outgoing_num[large_list[0]] > 0):
            outgoing_num[large_list[0]] = outgoing_num[large_list[0]] - 1
            heappop(large_list)

    return medians

## Schedule Tasks on Minimum Machines
'''
Given a set of n number of tasks, implement a task scheduler method, tasks(), to run in O(n logn) time that finds the minimum number of machines required to complete these n tasks.
Flow:
- Create a heap to store all incoming tasks in order of their start times.
- Create a second heap that will keep track of machines in use and the task that is currently using the machine.
- Loop through the list of tasks to schedule them on available machines.
- Take the task from the first heap with the earliest start time. Now, check whether there are any machines in use. If there are, compare the end time of the machine whose workload ends the earliest with the start time of the current task. If the machine is free at the start time of the current task, we can use it. Therefore, we update the scheduled end time of this machine in the second heap.
- If any conflicting task arises, allocate a new machine for this task. Otherwise, push it onto the existing machine.
- For tasks with similar start times, the task with an earlier end time will be preferred over the other.
- Keep the count of machines in use and return it when all tasks are processed.
O(nlogn) - O(n)
'''
def tasks(tasks_list):
    # to count the total number of machines for "optimal_machines" tasks
    optimal_machines = 0
    # empty list to store tasks finish time
    machines_available = []
    # converting list of set "optimal_machines" to a heap
    heapq.heapify(tasks_list)

    while tasks_list:  # looping through the tasks list
        # remove from "tasks_list" the task i with earliest start time
        task = heapq.heappop(tasks_list)

        if machines_available and task[0] >= machines_available[0][0]:
            # top element is deleted from "machines_available"
            machine_in_use = heapq.heappop(machines_available)

            # schedule task on the current machine
            machine_in_use = (task[1], machine_in_use[1])

        else:
            # if there's a conflicting task, increment the
            # counter for machines and store this task's
            # end time on heap "machines_available"
            optimal_machines += 1
            machine_in_use = (task[1], optimal_machines)

        heapq.heappush(machines_available, machine_in_use)

    # return the total number of machines used by "tasks_list" tasks
    return optimal_machines

### *** K-way Merge
'''
The k-way merge pattern helps to solve problems involving a list of sorted arrays.
Here is what the pattern looks like:
- Insert the first element of each array in a min-heap.
- Next, remove the smallest element from the heap and add it to the merged array.
- Keep track of which array each element comes from.
- Then, insert the next element of the same array into the heap.
- Repeat steps 2 to 4 to fill the merged array in sorted order.
'''

## Merge Sorted Array
'''
Given two sorted integer arrays, nums1 and nums2, and the number of data elements in each array, m and n, implement a function that merges the second array into the first one. You have to modify nums1 in place.
Flow:
Initialize two pointers, p1 and p2, that point to the last data elements in nums1 and nums2, respectively.
Initialize a pointer p, that points to the last element of nums1.
If the value at p1 is greater than the value at p2, set the value at p equal to p1 and decrement p1 and p by 1.
Else, if the value at p2 is greater than the value at p1, set the value at p equal to p2 and decrement p2 and p by 1.
Continue the traversal until nums2 is merged with nums1.
Naive approach: append the second list to the first—at a cost of  O(n) and then sort it using quick sort O((m + n)log(m + n))
Optimized approach: 
- Initialize two pointers that point to the last data elements in both arrays.
- Initialize a third pointer that points to the last index of nums1.
- Traverse nums1 from the end using the third pointer and compare the values corresponding to the first two pointers.
- Place the larger of the two values at the third pointer’s index.
- Repeat the process until the two arrays are merged.
O(m + n) - O(1)
'''
def merge_sorted(nums1, m, nums2, n):
    p1 = m - 1  
    p2 = n - 1 
    x = 0
    for p in range(n + m - 1, -1, -1):
        if p2 < 0:
            break
        x += 1
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
    return nums1

## Kth Smallest Number in M Sorted Lists
'''
Given m number of sorted lists in ascending order and an integer k, find the kth smallest number among all the given lists.
Flow:
- Push the first element of each list onto the min-heap.
- Pop the smallest number from the min-heap along with the list whose element is popped.
- If there are more elements in the list, increment the counter and push its next element onto the min-heap.
- Repeat the pop and push steps while there are elements in the min-heap, and only stop once you’ve removed k numbers from the heap.
- The last number popped from the heap during this process is the required kth smallest number.
Naive approach: A naive approach to this problem can be to merge all the arrays into a single array and then iterate over it to find the kth smallest number from it. O(nlogn)
Optimized approach: The key idea in this approach is to use a min-heap to allow us to efficiently find the smallest of a changing set of numbers.
- We first populate the min-heap with the smallest number from each of the m lists.
- Then, proceed to pop from the heap k times to reach the kth smallest number.
- Within a list, there may be numbers smaller than even the smallest number of another list. Because of this, every time we pop a number, we insert the next number from the same list. That way, we’re always considering the right candidates in the search for the kthsmallest number.
O((m+l)logm) - O(n)
'''
def k_smallest_number(lists, k):
    # storing the length of lists to use it in a loop later
    list_length = len(lists)
    # declaring a min-heap to keep track of smallest elements
    kth_smallest = []
    # to check if input lists are empty
    empty_list = []

    for index in range(list_length):
        # if there are no elements in the input lists, return []
        if lists[index] == empty_list:
            continue;
        else:
            # placing the first element of each list in the min-heap
            heappush(kth_smallest, (lists[index][0], 0, lists[index]))

    # set a counter to match if our kth element
    # equals to that counter, return that number
    numbers_checked, smallest_number = 0, 0
    while kth_smallest:  # iterating over the elements pushed in our min-heap
        # get the smallest number from top of heap and its corresponding list
        smallest_number, index, top_num_list = heappop(kth_smallest)
        numbers_checked += 1

        if numbers_checked == k:
            break

        # if there are more elements in list of the top element,
        # add the next element of that list to the min-heap
        if index + 1 < len(top_num_list):
            heappush(
                kth_smallest, (top_num_list[index + 1], index + 1, top_num_list))

    # return the Kth number found in input lists
    return smallest_number

## Find K Pairs with Smallest Sums
'''
Given two lists, and an integer k, find k pairs of numbers with the smallest sum so that in each pair, each list contributes one number to the pair.
Flow:
- Initialize a heap to store the sum of pairs and their respective indexes.
- Initially, we start making pairs by pairing only the first element of the second list with each element of the first list. We push the pairs onto a min-heap, sorted by the sum of each pair.
- Use another loop to pop the smallest pair from the min-heap, noting the sum of the pair and the list indexes of each element, and add the pair to a result list.
- To make new pairs, we move forward in the second list and pair the next element in it with each element of the first list, pushing each pair on the min-heap.
- We keep pushing and popping pairs from the min-heap until we have collected the required k smallest pairs in the result list.
Naive approach: One way to solve this problem is by creating all possible pairs from the given lists. Once we’ve created all pairs, sort them according to their sums, and remove the first k pairs from the list. O(m*n*log(m*n))
Optimized approach:
- We start by pairing only the first element of the second list with each element of the first list. The sum of each pair and their respective indexes from the lists, i and j, are stored on a min-heap.
- Next, we use a second loop in which at each step, we do the following:
  * We pop the pair with the smallest element from the min-heap and collect it in a result list.
  * We make a new pair in which the first element is the first element from the pair we just popped and the second element is the next element in the second list.
  * We push this pair along with its sum onto the min-heap.
  * We keep a count of the steps and stop when the min-heap becomes empty or when we have found k pairs.
O((m+k)logm) - O(m)
'''
def k_smallest_pairs(list1, list2, k):
    # storing the length of lists to use it in a loop later
    list_length = len(list1)
    # declaring a min-heap to keep track of the smallest sums
    min_heap = []
    # to store the pairs with smallest sums
    pairs = []

    # iterate over the length of list 1
    for i in range(min(k, list_length)):
        # computing sum of pairs all elements of list1 with first index
        # of list2 and placing it in the min-heap
        heappush(min_heap, (list1[i] + list2[0], i, 0))

    counter = 1

    # iterate over elements of min-heap and only go upto k
    while min_heap and counter <= k:
        # placing sum of the top element of min-heap
        # and its corresponding pairs in i and j
        sum_of_pairs, i, j = heappop(min_heap)

        # add pairs with the smallest sum in the new list
        pairs.append([list1[i], list2[j]])

        # increment the index for 2nd list, as we've
        # compared all possible pairs with the 1st index of list2
        next_element = j + 1

        # if next element is available for list2 then add it to the heap
        if len(list2) > next_element:
            heappush(min_heap,
                     (list1[i] + list2[next_element], i, next_element))

        counter += 1

    # return the pairs with the smallest sums
    return pairs

## Merge K Sorted Lists
'''
Given an array of k sorted linked lists, your task is to merge them into a single sorted list.
Flow:
- Traverse the input lists in pairs using head pointers.
- Compare the node values of lists in each pair and add the smaller one to a dummy list.
- Repeat the above steps until all the values from the lists in a pair are added.
- Compare this new list with the resultant list of the next pair.
Approach: use the divide and conquer technique, starting with pairing the lists and then merging each pair. We repeat this until all the given lists are merged. O(nlogk) - O(1)
'''
def merge_2_lists(head1, head2):  # helper function
    dummy = LinkedListNode(-1)
    prev = dummy  # set prev pointer to dummy node

    # traverse over the lists until both or one of them becomes null
    while head1 and head2:
        if head1.data <= head2.data:
            # if l1 value is <=  l2 value, add l1 node to the list
            prev.next = head1
            head1 = head1.next
        else:
            # if l1 value is greater than l2 value, add l2 node to the list
            prev.next = head2
            head2 = head2.next
        prev = prev.next

    if head1 is not None:
        prev.next = head1
    else:
        prev.next = head2

    return dummy.next

def merge_k_lists(lists):  # Main function
    if len(lists) > 0:
        step = 1
        while step < len(lists):
            # Traversing over the lists in pairs to merge with result
            for i in range(0, len(lists) - step, step * 2):
                lists[i].head = merge_2_lists(lists[i].head, lists[i + step].head)
            step *= 2
        return lists[0].head
    return

## Kth Smallest Element in a Sorted Matrix
'''
Find the kth smallest element in an (nxn), where each row and column of the matrix is sorted in ascending order.
Although there can be repeating values in the matrix, each element is considered unique and, therefore, contributes to calculating the kth smallest element.
Flow: 
- Push the first element of each row of the matrix in a min-heap.
- Remove the top (root) of the min-heap.
- If the popped element has the next element in its row, push that element to the heap.
- If k elements have been removed from the heap, return the last popped element.
Approach: A key observation when tackling this problem is that the matrix is sorted along rows and columns. This means that whether we look at the matrix as a collection of rows or as a collection of columns, we see a collection of sorted lists.
- We push the first element of each row of the matrix in the min-heap, storing each element along with its row and column index.
- Remove the top (root) of the min-heap.
- If the popped element has the next element in its row, push the next element in the heap.
- Repeat steps 2 and 3 as long as there are elements in the min-heap, and stop as soon as we’ve popped k elements from it.
- The last popped element in this process is the kth smallest element in the matrix.
O((n+k)logn) - O(n)
'''
def kth_smallest_element(matrix, k):
    # storing the number of rows in the matrix to use it in later
    row_count = len(matrix)
    # declaring a min-heap to keep track of smallest elements
    min_numbers = []
    
    for index in range(row_count):
        
        # pushing the first element of each row in the min-heap
        heappush(min_numbers, (matrix[index][0], index, 0))

    numbers_checked, smallest_element = 0, 0
    # iterating over the elements pushed in our min-heap
    while min_numbers:
        # get the smallest number from top of heap and its corresponding row and column
        smallest_element, row_index, col_index = heappop(min_numbers)
        numbers_checked += 1
        # when numbers_checked equals k, we'll return smallest_element
        if numbers_checked == k:
            break
        # if the current popped element has a next element in its row,
        # add the next element of that row to the min-heap
        if col_index + 1 < len(matrix[row_index]):
            heappush(min_numbers, (matrix[row_index][col_index + 1], row_index, col_index + 1))

    # return the Kth smallest element found in the matrix
    return smallest_element

### *** Top K Elements
'''
The best data structure to keep track of the smallest or largest k elements is heap. With this pattern, we either use a max-heap or a min-heap to find the smallest or largest k elements, respectively.
'''
## Kth Largest Element in a Stream
'''
Given an infinite stream of integers, nums, design a class to find the kth largest element in a stream.
Flow: 
- Create a min heap from the elements in the input number stream
- Pop from the heap until the its size=k
- In the add function, push the incoming number to the stream
- If the size exceeds k, pop from the heap
- After iterating all of the numbers, return the top of the heap as the kth largest element
Naive approach: The naive solution is first to sort the data and then find kth largest element. Insertion sort is an algorithm that can be used to sort the data as it appears. O(n^2) - O(1)
Optimized approach: 
- Use a min-heap to store the numbers in the input stream.
- If the size of the heap > k,pop from the heap.
- Return the top element of the min-heap.
Construct: O(nlogn) - Add: O(logk) - O(1)
O(n)
'''
import heapq
class KthLargest:
    # constructor to initialize heap and add values in it
    def __init__(self, k, nums):
        self.k = k
        self.top_k_heap = nums
        heapq.heapify(self.top_k_heap)
        while len(self.top_k_heap) > k:
            heapq.heappop(self.top_k_heap)
            
    # adds element in the heap
    def add(self, val):
        heapq.heappush(self.top_k_heap, val)
        if len(self.top_k_heap) > self.k:
            heapq.heappop(self.top_k_heap)
        return self.return_Kth_largest()
        
    # returns kth largest element from heap
    def return_Kth_largest(self):
        return self.top_k_heap[0]

## Reorganize String
'''
Given a string, rearrange it so that any two adjacent characters are not the same. If such a reorganization of the characters is possible, output any possible valid arrangement. Otherwise, return an empty string.
Flow:
- Store each character and its frequency in a hash map.
- Construct a max-heap using the character frequency data, so that the most frequently occurring character is at the root of the heap.
- Iterate over the heap and in each iteration, pop the most frequently occurring character and append it to the result string.
- Decrement the frequency of the popped character (as we have consumed one occurrence of it) and push it back onto the heap if the updated frequency is greater than 0.
- Return the result string when the heap becomes empty.
Naive approach: The naive approach is to generate all possible permutations of the given string and check if the generated string is a valid arrangement or not. O(n^2) - O(1)
Optimized solution: 
- Store each character and its frequency in a hash map.
- Construct a max-heap using the character frequency data so that the most frequently occurring character is at the root of the heap.
- Iterate over the heap until all the characters have been considered.
  * Pop the most frequently occurring character from the heap and append it to the result string.
  * Decrement the count of the character (as we have used one occurrence of it), and push the character that was popped in the previous iteration back onto the heap.
  * If the frequency count of the character that was just popped remains more than 0, save it for use in the next iteration.
O(nlogc) (c alphabet constant) --> O(n) - O(1)
'''
from collections import Counter
def reorganize_string(input_string):
    
    char_counter = Counter(input_string)
    most_freq_chars = []

    for char, count in char_counter.items():
        most_freq_chars.append([-count, char])

    heapq.heapify(most_freq_chars)

    previous = None
    result = ""

    while len(most_freq_chars) > 0 or previous:

        if previous and len(most_freq_chars) == 0:
            return ""

        count, char = heapq.heappop(most_freq_chars)
        result = result + char
        count = count + 1

        if previous:
            heapq.heappush(most_freq_chars, previous)
            previous = None

        if count != 0:
            previous = [count, char]

    return result + ""

## K Closest Points to Origin
'''
Given a list of points on a plane, where the plane is a 2-D array with (x, y) coordinates, find the k closest points to the origin (0,0).
Flow: 
- Push the first k points to the heap.
- Compare the distance of the point with the distance of the top of the heap.
- Push and pop the point from the heap.
- Return the points from the heap.
Naive approach: When thinking about how to solve this problem, it may help to solve a simpler problem—find the point closest to the origin. This would involve a linear scan through the unsorted list of points, with, at each step, a comparison between the closest point discovered so far and the current point from the list. O(nk)
Optimized approach: O(nlogk) - O(k)
'''
class Point:
    # __init__ will be used to make a Point type object
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # __lt__ is used for max-heap
    def __lt__(self, other):
        return self.distance_from_origin() > other.distance_from_origin()

    # __str__ is used to print the x and y values
    def __str__(self):
        return '[{self.x}, {self.y}]'.format(self=self)

    # distance_from_origin calculates the distance using x, y coordinates
    def distance_from_origin(self):
        # ignoring sqrt to calculate the distance
        return (self.x * self.x) + (self.y * self.y)

    __repr__ = __str__

def k_closest(points, k):
    points_max_heap = []

    # put first 'k' points in the max heap
    for i in range(k):
        heapq.heappush(points_max_heap, points[i])

    # go through the remaining points of the input array, if a 
    # point is closer to the origin than the top point of the 
    # max-heap, remove the top point from heap and add the point 
    # from the input array
    for i in range(k, len(points)):
        if points[i].distance_from_origin() \
         < points_max_heap[0].distance_from_origin():
            heapq.heappop(points_max_heap)
            heapq.heappush(points_max_heap, points[i])

    # the heap has 'k' points closest to the origin, return 
    # them in a list
    return list(points_max_heap)

## Top K Frequent Elements
'''
Given an array of integers arr and an integer k, return the k most frequent elements.
Flow:
- Create a min-heap of size k.
- Find the frequency of all elements and save results in a hash map with the frequency as the key and the numbers as the value.
- Insert key value pairs into the heap until the heap becomes full.
- Once the heap is full, remove the minimum element and insert a new element.
- Repeat until all numbers have been processed.
- The resulting heap contains the top K elements.
Naive approach#
The naive approach: build a map of elements and frequency by traversing the given array. After storing the elements and their respective frequencies in a map, sort the entries of the map according to the decreasing order of frequency. O(nlogn) - O(n)
Optimized approach: O(nlogk) or O(nlogn) if k == n - O(n + k)
'''
def top_k_frequent(arr, k):

    # find the frequency of each number
    num_frequency_map = {}
    for num in arr:
        num_frequency_map[num] = num_frequency_map.get(num, 0) + 1
    top_k_elements = []

    # go through all numbers of the num_frequency_map
    # and push them in the top_k_elements, which will have
    # top k frequent numbers. If the heap size is more than k,
    # we remove the smallest(top) number
    for num, frequency in num_frequency_map.items():
        heappush(top_k_elements, (frequency, num))
        if len(top_k_elements) > k:
            heappop(top_k_elements)

    # create a list of top k numbers
    top_numbers = []
    while top_k_elements:
        top_numbers.append(heappop(top_k_elements)[1])

    return top_numbers



### *** Practice
## 2 pointers - Valid Palindrome II
'''
Write a function that takes a string as input and checks whether it can be a valid palindrome by removing at most one character from it.
Flow: 
- Initialize two pointers at opposite ends of the string.
- If the values at the left and right indexes match, move both toward the middle until they meet.
- If a mismatch occurs, skip one of the elements and check the rest of the string for a palindrome.
- Skip the other element, and check for the palindrome.
- If no palindrome is obtained, return False, else if no more than one mismatch occurs throughout the traversal, return True.
'''
def helper(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True


def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return helper(s[left + 1:right + 1]) or helper(s[left:right])

        left += 1
        right -= 1

    return True

## SLiding Window - Best Time to Buy and Sell Stock
'''
Given an array where the element at the index i represents the price of a stock on day i, find the maximum profit that you can gain by buying the stock once and then selling it.
Flow: 
- Initialize buy and sell pointers to 0 and 1, respectively, and set maximum profit variable to 0.
- Iterate sell pointer over the prices, computing the current profit by subtracting the prices[buy] from the prices[sell].
- If prices[buy] is less than prices[sell], choose the maximum value from current profit or maximum profit and store it in the maximum profit. Otherwise, update buy to be equal to sell.
- Once the sell pointer reaches the end, return the maximum profit.
'''
## Merge Intervals - Meeting Rooms II
'''
We are given an input array of meeting time intervals, intervals, where each interval has a start time and an end time. Your task is to find the minimum number of meeting rooms required to hold these meetings.
Flow:
- Sort the given input intervals with respect to their start times.
- Initialize a min-heap and push the end time of the first interval onto it.
- Loop over the remaining intervals.
- In each iteration, compare the start time of the current interval with all the end times present in the heap.
- If the earliest end time of all intervals seen so far (the root of the heap) occurs before the start time of the current interval, remove the earliest interval from the heap and push the current interval onto the heap.
- Otherwise, allot a new meeting room, that is, add the current interval in the heap without removing any existing interval.
- After processing all the intervals, the size of the heap is the count of meeting rooms needed to hold the meetings.
'''

## In-place Reversal of a Linked List - Swap Nodes in Pairs
'''
Given a singly linked list, swap every two adjacent nodes of the linked list. After the swap, return the head of the linked list.
- Check to make sure that there are at least 2 nodes in the linked list.
- Swap the two nodes.
- Reconnect the swapped pair of nodes with the rest of the linked list.
- Repeat the process until only one node is left or we reach the end of the linked list.
'''

## K-way merge - Median of Two Sorted Arrays
'''
You’re given two sorted integer arrays, nums1 and nums2, of size m and n, respectively. Your task is to return the median of the two sorted arrays.
Flow:
- Pick the middle element of the longer array as the partition location. Let’s call this element left long. Let’s call the very next element right long.
- Figure out how many elements of the shorter array need to be in the first half. Let’s call this element left short. Let’s call the next element right short.
- Check Partition: If left long is less than right short and left short is less than right long, we can calculate the median of the two sorted arrays as the mean of Max(left long, left short) and Min(right long, right short).
- Otherwise, if left long is greater than right short, move left long to halfway between its current position and the start of the longer array, and update right long, left short and right short accordingly.
- Otherwise, if left short is greater than right long, move left short to halfway between its current position and the start of the shorter array, then figure out left long and update right short and right long accordingly.
- Repeat the Check Partition step and either return the median or move the partition.
'''