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