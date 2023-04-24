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
