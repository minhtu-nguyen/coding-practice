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

## Kth Largest Element in an Array
'''
Find the kth largest element in an unsorted array.
Flow: 
- Insert the first k numbers into the min-heap.
- Iterate the complete array.
- If any number is greater than the root, take the root out and insert the greater number.
- After iterating all of the numbers, return the root number as the kth largest element.
O((n-k)logk) - O(k)
'''
def find_kth_largest(array, k):
    k_numbers_min_heap = []
    # Put first k elements in the min heap
    for i in range(k):
        k_numbers_min_heap.append(array[i])
        # heappush(min_heap, array[i])
    heapq.heapify(k_numbers_min_heap)
    # Go through the remaining elements of the list, if the element from the list is greater than the
    # top(smallest) element of the heap, remove the top element from heap and add the element from array
    for i in range(k, len(array)):
        if array[i] > k_numbers_min_heap[0]:
            heapq.heappop(k_numbers_min_heap)
            heapq.heappush(k_numbers_min_heap, array[i])
    # The root of the heap has the Kth largest element
    return k_numbers_min_heap[0]

### *** Modified Binary Search
## Search in Rotated Sorted Array
'''
Given a sorted integer array, nums, and an integer value, target, the array is rotated by some arbitrary number. Search and return the index of target in this array. If the target does not exist, return -1.
Flow:
- Declare two low and high pointers that will initially point to the first and last indexes of the array, respectively.
- Declare a middle pointer that will initially point to the middle index of the array. This divides the array into two halves.
- Check if the target is present at the position of the middle pointer. If it is, return its index.
- If the first half of the array is sorted and the target lies in this range, update the high pointer to mid in order to search in the first half.
- Else, if the second half of the array is sorted and the target lies in this range, update the low pointer to mid in order to search in the second half.
- If the high pointer becomes greater or equal to the low pointer and we still haven’t found the target, return -1.
Naive approach: traverse the whole array while searching for our target. O(n) - O(1)
Optimized approach: O(logn) - O(1) iter | O(logn) recur
'''
def binary_search_rotated(nums, target):
    low = 0
    high = len(nums) - 1

    if low > high:
        return -1

    while low <= high:
        mid = low + (high - low) // 2
        if nums[mid] == target:
            return mid

        if nums[low] <= nums[mid]:
            if nums[low] <= target and target < nums[mid]:
                high = mid - 1 
            else:
                low = mid + 1 
        else:
            if nums[mid] < target and target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1 
    return -1

def binary_search(nums, low, high, target):

    if (low > high):
        return -1
    mid = low + (high - low) // 2

    if nums[mid] == target:
        return mid

    if nums[low] <= nums[mid]:
        if nums[low] <= target and target < nums[mid]:
            return binary_search(nums, low, mid-1, target)
        else:
            return binary_search(nums, mid+1, high, target)
    else:
        if nums[mid] < target and target <= nums[high]:
            return binary_search(nums, mid+1, high, target)
        else:
            return binary_search(nums, low, mid-1, target)


def binary_search_rotated(nums, target):
    return binary_search(nums, 0, len(nums)-1, target)

## First Bad Version
'''
The latest version of a software product fails the quality check. Since each version is developed upon the previous one, all the versions created after a bad version are also considered bad.
Suppose you have n versions with the IDs [1,2,...,n], and you have access to an API function that returns TRUE if the argument is the ID of a bad version.
Your task is to find the first bad version, which is causing all the later ones to be bad. You have to implement a solution with the minimum number of API calls.
Flow: 
- Initialize first to 1 and last to n.
- Calculate mid as the mean of 1 and n, and call the API function with mid. Increment the counter for the number of API calls.
- If the API function returns TRUE, set last to mid.
- Else, if the API function returns FALSE, set first to mid+1.
- While first < last, repeat the steps to adjust first and last, to recalculate mid, and to call the API function.
- Return the tuple containing the first bad version and the number of API calls.
Naive approach: The naive approach is to find the first bad version in the versions list by linear search. We traverse the whole version list one element at a time until we find the target version. O(n)
Optimized approach: O(logn) - O(1)
'''
def is_bad_version(n):
    pass
def first_bad_version(n):
    first = 1
    last = n
    calls = 0
    while first < last:  
        mid = first + (last - first) // 2
        if is_bad_version(mid):  #API call
            last = mid
        else:
            first = mid + 1
        calls += 1
    return first, calls

## Random Pick with Weight
'''
You’re given an array of positive integers, w, where w[i] describes the weight of the ith index.
You need to perform weighted random selection to return an index from the w array. The larger the value of w[i], the heavier the weight is. Hence, the higher the chances of its index being picked.
Flow:
- Generate a list of running sums from the given list of weights.
- Generate a random number, where the range for the random number will be from 0 to the maximum number in the list of running sums.
- Use binary search to find the index of the first running sum that is greater than the random value.
- Return the index found.
Naive approach: In order to correctly add bias to the randomized pick, we need to generate a probability line where the length of each segment is determined by the corresponding weight in the input array. Positions with greater weights have longer segments on the probability line, and positions with smaller weights have smaller segments. We can represent this probability line as a list of running sums of weights. 
O(n) - O(n) contruct - O(logn) - O(1) search
'''
import random
class RandomPickWithWeight:
    # Constructor

    def __init__(self, w):
        self.cum_sums = []
        cum_sum = 0
        # Calculating the weights running sums list
        for weight in w:
            cum_sum += weight
            self.cum_sums.append(cum_sum)
        self.total_sum = cum_sum

    def pick_index(self):
        target = random.randint(0, self.cum_sums[-1])

        # Assigning low pointer at the start of the array
        low = 0
        # Assigning high pointer at the end of the array
        high = len(self.cum_sums)
        # Binary search to find the target
        while low < high:
            mid = low + (high - low) // 2
            if target > self.cum_sums[mid]:
                low = mid + 1
            else:
                high = mid
        return low

## Find K Closest Elements
'''
Given a sorted integer array nums and two integers—k and num—return the k closest integers to num in this array. Ensure that the result is sorted in ascending order.
Flow: 
- Use binary search to find the closest elements to num and assign a left pointer to the closest number that’s less than num and right to left + 1.
- If the right value is closer to num than the left value, move the right pointer one step forward.
- Else, move the left pointer one step backwards.
- When the window size becomes equal to k, return the closest elements starting from the left pointer towards the right pointer.
Naive approach: The naive approach is to compute the distance of every element from the given num. Then, we'll sort the elements in ascending order based on the computed distances and store them in a new array. O(nlogn) - O(n)
Optimized approach: We use binary search to locate either num in the array or the number nearest to it. We'll then set left to point to the number closest to and less than num, and right to num itself (or, if num does not exist in the array, then the number in the array closest to it). We'll then move left backward and right forward until the elements between them are the k elements closest to num.
O(logn + k) - O(1)
'''
def binary_search(array, target):
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] == target:
            return mid
        if array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

def find_closest_elements(nums, k, num):

    # If the length of the array is same as k,
    # return the original array
    if len(nums) == k:
        return nums

    # Do a binary search to find the element closest to num
    # and initialize two pointers for the sliding window

    left = binary_search(nums, num) - 1
    right = left + 1

    # While the sliding window's size is less than k
    while right - left - 1 < k:
        # check for out of bounds
        if left == -1:
            right += 1
            continue

        # Expand the window towards the side with the closer number
        # Be careful to not go out of bounds with the pointers
        # |a - num| < |b - num|,
        # |a - num| == |b - num|
        if right == len(nums) or \
                abs(nums[left] - num) <= abs(nums[right] - num):
            left -= 1
        else:
            right += 1

    # Return the window
    return nums[left + 1:right]

## Single Element in a Sorted Array
'''
In this problem, you’re given an array of sorted integers in which all of the integers, except one, appears twice. Your task is to find the single integer that appears only once.
The solution should have a time complexity of O(logn) or better and a space complexity of O(1).
Flow:
- Initialize the left, mid and right variables.
- Calculate mid. Decrement it by one if it’s an odd index.
- If mid and mid+1 are not the same number, then change right to point to mid and calculate mid again.
- If mid and mid+1 are the same, then change left to mid+2.
- Repeat the steps until the left pointer becomes greater than the right pointer and calculate the midpoint in each step.
- Return the element at the left index.
O(logn) - O(1)
'''
def single_non_duplicate(nums): 
    l = 0
    r = len(nums) - 1
    while l < r: # Calculating mid point
        mid = l + (r - l) // 2
        if mid % 2 == 1: # If mid is odd, decrement it
            mid -= 1     # to  the preceding even index
            # If the element at mid and mid + 1 are the same then
            # the single element must appear after the mid point
        if nums[mid] == nums[mid + 1]: 
            l = mid + 2
        else :# Otherwise we must search before the mid point
            r = mid
    return nums[l]

### *** Subsets
'''
The subsets pattern is useful in finding the permutations and combinations of elements in a data structure.
The idea is to consider the data structure as a set and make a series of subsets from this set. The subsets generated are based on certain conditions that the problem provides us.
'''

## Subsets
'''
Given a set of integers, find all possible subsets within the set.
Flow:
- Compute the number of possible subsets of the given set using 2^n, where n is the number of elements.
- Start a loop from 0 to the count of subsets and add an empty list to the results list in the first iteration.
- In each iteration, create a bit mask of length n for each element in the input set. If the ith bit is set, set[i] will be present in the current subset.
- Add the current subset to the list of subsets.
Naive approach: The naive approach is to use the cascading algorithm to find all subsets of a set. The idea of this approach is to iterate the set and append each element to the existing subsets as new subsets. Therefore, with each iteration, the size of the subsets doubles. O(n*2^n) - O(n*2^n)
Optimized approach: O((2^n)*n) - O(n)
'''
def get_bit(num, bit):
    temp = (1 << bit)
    temp = temp & num
    if temp == 0:
        return 0
    return 1

def find_all_subsets(v):
    sets = []
    
    if not v:
        return [[]]
    else:
        subsets_count = 2 ** len(v)
        for i in range(0, subsets_count):
            st = set()
            for j in range(0, len(v)):
                if get_bit(i, j) == 1 and v[j] not in st:
                    st.add(v[j])
            
            if i == 0:
                sets.append([])
            else:
                sets.append(list(st))
    return sets

## Permutations
'''
Given an input string, return all possible permutations of the string.
Flow:
- Starting from the first index as the current index, recursively compute the permutations of the input string.
- Compute the permutation by swapping the current index with every index in the remaining string.
- Recurse the computation step by incrementing the current index by 1.
- If we reach the end of the string, store the current string as a permutation.
- Return the list of all permutations.
O(n!) - O(n)
'''
# This function will swap characters for every permutation
def swap_char(word, i, j):
    swap_index = list(word)
    swap_index[i], swap_index[j] = swap_index[j], swap_index[i]

    return ''.join(swap_index)


def permute_string_rec(word, current_index, result):
    # Prevents adding duplicate permutations
    if current_index == len(word) - 1:
        result.append(word)

        return

    for i in range(current_index, len(word)):
        # swaps character for each permutation
        swapped_str = swap_char(word, current_index, i)

        # recursively calls itslef to find each permuation
        permute_string_rec(swapped_str, current_index + 1, result)

def permute_word(word):
    result = []

    # Starts finding permuations from start of string
    permute_string_rec(word, 0, result)

    return result

## Letter Combinations of a Phone Number
'''
Given a string having digits from 2-9 inclusive, return all the possible letter combinations that can be made from the numbers in the string. Return the answer in any order.
Flow:
- Initialize a dictionary that maps the digits to their characters.
- Create a backtracking function that considers a digit as starting point and generates all possible combinations with that letter.
- If the length of our combination is the same as the input, we have an answer. Add it to the list of results and backtrack.
- Otherwise, find all the possible combinations of characters that correspond to the current digit.
Approach:
- We return an empty array if the input is an empty string.
- We initialize a data structure (for example, a dictionary) that maps the digits to their characters. For example, we map 2 to a, b, and c.
- The following parameters will be passed to our backtracking function: the path of the current combination of characters, the index of the given digit array, digits, letters, and combinations.
- If our current combination of characters is the same length as the input, then we have an answer. Therefore, add it to our list of results and backtrack.
- Otherwise, we get all the characters that correspond to the current digit we are looking at: digits[index].
- We then loop through these characters. We add each character to our current path and call backtrack again but move on to the next digit by incrementing the index by 1. To avoid using the same character twice in a combination, we make sure to remove the character from the path once we’ve made its combinations.
O(k^n*n) - O(n*k)
'''
# Use backtrack function to generate all possible combinations
def backtrack(index, path, digits, letters, combinations):
    # If the length of path and digits is same,
    # we have a complete combination
    if len(path) == len(digits):
        s = ""
        s = s.join(path)
        combinations.append(s)
        return  # Backtrack
    # Get the list of letters using the index and digits[index]
    possible_letters = letters[digits[index]]
    if possible_letters:
        for letter in possible_letters:
            # Add the letter to our current path
            path.append(letter)
            # Move on to the next category
            backtrack(index + 1, path, digits, letters, combinations)
            # Backtrack by removing the letter before moving onto the next
            path.pop()

def letter_combinations(digits):
    combinations = []
    
    # If the input is empty, immediately return an empty answer array
    if len(digits) == 0:
        return []

    #  Mapping the digits to their corresponding letters
    digits_mapping = {
        "1": [""],
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"]}

    # Initiate backtracking with an empty path and starting index of 0

    backtrack(0, [], digits, digits_mapping, combinations)
    return combinations

## Generate Parentheses
'''
For a given number, n, generate all combinations of balanced parentheses.
Flow:
- Create an output list to store all the valid combinations of parantheses.
- Call the back_track function with initial parameters set to n, empty string and 0 for the count of both, opening and closing parantheses.
- If the count of opening and closing parentheses equals n, then we have a valid combination of parentheses, so we will append the string to our final list.
- Otherwise, we will check if the number of opening parentheses is less than n. If yes, we will add opening parentheses to our string and increment the count.
- Lastly, we will check the count of closing parentheses. If the count is less than the count of opening parentheses, then we will add closing parentheses to our string and increment its count.
O(4^n) - O(n)
'''
def back_track(n, left_count, right_count, output, result):
    # Base case where count of left and right braces is "n"
    if left_count >= n and right_count >= n:
        output_str = str(output)

        replace_str = "[],' "

        for c in replace_str:
            output_str = output_str.replace(c, "")
        result.append(output_str)

    # Case where we can still append left braces
    if left_count < n:
        output.append('(')
        back_track(n, left_count + 1,
                   right_count, output, result)
        output.pop()

    # Case where we append right braces if the current count
    # of right braces is less than the count of left braces
    if right_count < left_count:
        output.append(')')
        back_track(n, left_count,
                   right_count + 1, output, result)
        output.pop()

def generate_combinations(n):
    result = []
    output = []
    back_track(n, 0, 0, output, result)

    return result

### *** Greedy Techniques
'''
Greedy is an algorithmic paradigm that builds up a solution piece by piece. This means it chooses the next piece that offers the most obvious and immediate benefit. A greedy algorithm, as the name implies, always makes the choice that seems to be the best at the time. It makes a locally-optimal choice in the hope that it will lead to a globally optimal solution. In other words, greedy algorithms are used to solve optimization problems.

Greedy algorithms work by recursively constructing a solution from the smallest possible constituent parts. A recursion is an approach to problem-solving in which the solution to a particular problem depends on solutions to smaller instances of the same problem. While this technique might seem to result in the best solution, greedy algorithms have the downside of getting stuck in local optima and generally do not return the global best solution. There are a number of problems that use the greedy technique to find the solution, especially in the networking domain, where this approach is used to solve problems such as the traveling salesman problem and Prim’s minimum spanning tree algorithm.
'''
## Jump Game I
'''
In a single-player jump game, the player starts at one end of a series of squares, with the goal of reaching the last square.
At each turn, the player can take up to s steps towards the last square, where s is the value of the current square.
For example, if the value of the current square is 3, the player can take either 3 steps, or 2 steps, or 1 step in the direction of the last square. The player cannot move in the opposite direction, that is, away from the last square.
You have been tasked with writing a function to validate whether a player can win a given game or not.
Flow:
- Set the last element in the array as your initial target.
- Traverse the array from the end to the first element in the array.
- If the current index is reachable from any preceding index, based on the value at that index, make that index the new target.
- If you reach the first index of the array without finding any index from which the current target is reachable, return FALSE.
- Else, if you are able to move each current target backwards all the way to the first index of the array, you’ve found a path from the start to the end of the array. Return TRUE.
Naive approach: The naive approach would be to check if we can reach the end of the array from every single element until it returns TRUE or if we reach the end of the array itself. 
Optimized approach: We set the target index as the last index of the array. We then traverse the array backwards and verify if we can reach the target index from any of the previous indexes. If we're able to reach it, we update the target index with the index that allows us to jump to the target index. We repeat this process until we've traversed the entire array. We return TRUE if, through this process, we are able to reach the first index of the array; otherwise, we return FALSE. O(n) - O(1).
'''
def jump_game(nums):
    target_num_index = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        if target_num_index <= i + nums[i]:
            target_num_index = i

    if target_num_index == 0:
        return True
    return False

## Boats to Save People
'''
You’re given an array, people, where people[i] is the weight of the ith person, and an infinite number of boats, where each boat can carry a maximum weight, limit. Each boat carries, at most, two people at the same time. This is provided that the sum of the weight of those people is under or equal to the weight limit.
You need to return the minimum number of boats to carry every person in the array.
Flow:
- Sort the people array and initialize three variables to mark the heaviest and lightest person in the list and to calculate the number of boats needed.
- Since the array is already sorted, point the lightest variable to the first element in our people array, and the heaviest variable to the last element in our array.
- Check if the combined weight of the lightest and heaviest person is under the weight limit. If it is, then increment the lightest variable by one, decrement the heaviest variable by one, and increment the boats used by one. If it isn’t, then only decrement the heaviest pointer by one and increment the boat count by one.
- Return the answer variable as our final output.
Naive approach: The naive approach would first count the number of people whose weight is equal to the boat's limit. For the remaining people, we will pair each person with every other person in the array such that their combined weight is less than or equal to the boat's limit. Once we have all such pairs, we will select the pair with the maximum weight and increment the boat count by 1. We will do the same for all the remaining people until all of them have been placed in a boat. O(n^2)
Optimized approach: We decided that if the person with the highest weight could share the boat with the person with the lowest weight, we would always make them share. If not, the person with the highest weight is not able to be paired with anyone and must instead get a boat of their own. By implementing the greedy pattern we traverse the array forward and backward at the same time with the help of two pointers, indicating the heaviest and lightest person currently on the ship. We calculated the sum of both their weights and adjusted our pointers accordingly until the pointers crossed each other. O(nlogn) - O(n) sorting
'''
def rescue_boats(people, limit):
    # Sorting the peoples array.
    people.sort()
    # After the array is sorted, the person with the lightest weight will be at the start of the list.
    lightest = 0
    # The heaviest person will be at the end of the list
    heaviest = len(people) - 1
    answer = 0  # The number of boats used before we save anyone
    while lightest <= heaviest:  # The condition to share boats.
        answer += 1  # We increase the number of boats we used by one
        # If there is still space for another person than we increment the light pointer by one too.
        if people[lightest] + people[heaviest] <= limit:
            lightest += 1  # Moving forward because we added the lightest person too
        # The heaviest person will always get a boat, hence we would always shift to the 2nd person with most weight.
        heaviest -= 1
    return answer  # Returning the minimum number of boats

## Gas Stations 
'''
There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
We have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i+1)th station. We begin the journey with an empty tank at one of the gas stations.
Given two integer arrays, gas and cost, return the starting gas station’s index if we can travel around the circuit once in the clockwise direction. Otherwise, return −1.
Flow:
- Calculate the total gas and total cost from the arrays. If the total gas is less than the total cost, return -1, since we can never complete a loop from any index.
- Set the starting index and total gas available equal to 0.
- Traverse the gas array. During this traversal, subtract the current element in the gas array with the cooresponding index in the cost array, and add this value to the total gas.
- If at any point, the total gas becomes less than 0, that means, we can not complete a loop form this starting index, so we increment the starting index by 1.
- Return the starting index at the end of the traversal.
Naive approach: The naive approach to this would be to choose each station as a starting point and try to visit every other station while maintaining the fuel level. This will determine whether or not it is possible to visit the next station. If not, we choose another starting station and try to visit every other station in the array until we reach the starting station again. O(n^2)
Optimized approach: O(n) - O(1)
'''
def gas_station_journey(gas, cost):
    if sum(cost) > sum(gas):  # If the sum of the costs is greater than the gas
        return -1             # available to us, then we can't ever reach the same point again.

    # Setting the indexes that we will start our analysis from
    total_gas, starting_index = 0, 0
    for i in range(len(gas)):  # Traversing the array from start to finish
        # Checking how much gas we have left in the tank after going to
        # the next station.
        total_gas = total_gas + (gas[i] - cost[i])
        # the next station.

        if total_gas < 0:
            total_gas = 0
            starting_index = i + 1

    return starting_index

## Two City Scheduling
'''
A company is planning to interview 2n people. You're given the array costs where costs[i]=[aCost i ​ ,bCost i ​ ] . The cost of flying the ith person to city  A is aCost i ​ , and the cost of flying the ith person to city  B is bCosti ​ . Return the minimum cost to fly every person to a city such that exactly n n people arrive in each city.
Flow:
- Initialize an array to store the extra amount each person would take to travel to city A compared to city B.
- In the same array, along with the difference, also append the cost each person takes to travel to city A and city B.
- Sort the array in ascending order based on the difference between the cost of travelling to city A and city B.
- Calculate the minimum possible cost to send people evenly among the two cities.
O(nlogn) - O(m+n)
'''
# We will be using math.floor for floor division in this problem.
import math


def two_city_scheduling(costs):
    # Array that we will use to divide the group in two equal parts.
    difference = []
    # Initiliazing a new variable to calculate the minimum cost required to send exactly n people to both cities.
    answer = 0
    # Initiliazing a loop, to calculate the difference to travel to City A or B.
    for cost1, cost2 in costs:
        # Calculating the differences, and adding costs for City A and B alongside in an array.
        difference.append([cost1 - cost2, cost1, cost2])
    # We sort the array based on the differences we calculated above.
    difference.sort()
    length_diff = len(difference)
    # Initiliazing a loop, to add the relevent costs to our answer variable.
    for i in range(length_diff):
        if i < math.floor(length_diff / 2):
            # We will send this half to City A
            answer = answer + difference[i][1]
        else:
            # We will send this half to City B
            answer = answer + difference[i][2]
    return answer

## Minimum Number of Refueling Stops
'''
You need to find the minimum number of refueling stops that a car needs to make to cover a distance, target. For simplicity, assume that the car has to travel from west to east in a straight line. There are various fuel stations on the way that are represented as a 2-D array of stations, i.e., stations[i] = =[d i ​ ,f i ​ ] , where � � d i ​ is the distance (in miles) of the ith gas station from the starting position, and fi is the amount of fuel (in liters) that it stores. Initially, the car starts with k liters of fuel. The car consumes one liter of fuel for every mile traveled. Upon reaching a gas station, the car can stop and refuel using all the petrol stored at the station. In case it cannot reach the target, the program simply returns −1.
Flow:
- If the start fuel is greater than or equal to the target, then the car doesn’t need to refuel, so return 0.
- Iterate over the refueling stations until the maximum distance is less than the target and the car is not out of fuel.
- If the car can reach the next station from the current position, then add its fuel capacity to the max-heap.
- If the car cannot reach the next fuel station, pop the station with the highest fuel value from the max-heap, add its fuel to the car’s tank, and increment the stops.
- Return the number of stops. If the car cannot reach the destination even after stopping at all the fuel stations, return −1.
O(nlogn) - O(n)
'''
def min_refuel_stops(target, start_fuel, stations):
    # If starting fuel is already greater or equal to target, no need to refuel
    if start_fuel >= target:
        return 0
    # Create a max heap to store the fuel capacities of stations in
    # such a way that maximum fuel capacity is at the top of the heap
    max_heap = []
    # Initialize variables for loop
    i, n = 0, len(stations)
    stops = 0
    max_distance = start_fuel
    # Loop until the car reach the target or the car is out of fuel
    while max_distance < target:
        # If there are still stations and the next one is within range, add its fuel capacity to the max heap
        if i < n and stations[i][0] <= max_distance:
            heapq.heappush(max_heap, -stations[i][1])
            i += 1
        # If there are no more stations and we can't reach the target, return -1
        elif not max_heap:
            return -1
        # Otherwise, fill up at the station with the highest fuel capacity and increment stops
        else:
            max_distance += -heapq.heappop(max_heap)
            stops += 1
    # Return the number of stops taken
    return stops

### *** Backtracking
'''
Backtracking is different from recursion because, in recursion, the function calls itself until it reaches a base case whereas backtracking tries to explore all possible paths to a solution.
The way backtracking works is that it first explores one possible option. If the required criteria have been met with that option, we choose a path that stems from that option and keep on exploring that path. If a solution is reached from this path, we return this solution. Otherwise, if a condition is violated from the required set of conditions, we backtrack and explore another path.
'''
## N-Queens
'''
Given a chessboard of size n×n, determine how many ways n queens can be placed on the board, such that no two queens attack each other.
A queen can move horizontally, vertically, and diagonally on a chessboard. One queen can be attacked by another queen if both share the same row, column, or diagonal.
Flow:
- Start by placing a queen anywhere in the first row of a chess board.
- Since no other queen may be placed in a row that already has a queen, search for a safe position for the next queen in the next row.
- Iterate over the rows to find a safe placement for the queens. Store the column number where a queen is placed in a list.
- If a safe position is not found, backtrack to the previous valid placement. Search for another solution.
- If a complete solution is found, add it to the results array, and backtrack to find other valid solutions in the same way.
Naive solution: we could find all configurations with all possible placements of n queens and then determine for every configuration if it is valid or not. O((n^2/n)) - O(n)
Optimized solution: O(n^n) - O(n)
'''
# This method determines if a queen can be placed at proposed_row, proposed_col
# with current solution i.e. this move is valid only if no queen in current
# solution may attack the square at proposed_row and proposed_col
def is_valid_move(proposed_row, proposed_col, solution):
    for i in range(0, proposed_row):
        old_row = i
        old_col = solution[i]
        diagonal_offset = proposed_row - old_row
        if (old_col == proposed_col or
            old_col == proposed_col - diagonal_offset or
                old_col == proposed_col + diagonal_offset):
            return False
            
    return True

# Recursive worker function
def solve_n_queens_rec(n, solution, row, results):
    if row == n:
        results.append(solution)
        return

    for i in range(0, n):
        valid = is_valid_move(row, i, solution)
        if valid:
            solution[row] = i
            solve_n_queens_rec(n, solution, row + 1, results)

# Function to solve N-Queens problem
def solve_n_queens(n):
    results = []
    solution = [-1] * n
    solve_n_queens_rec(n, solution, 0, results)
    return len(results)

# This solution uses stack to store the solution.
# Stack will hold only the column values and one solution
# will be stored in the stack at a time.

def is_valid_move(proposed_row, proposed_col, solution):
  # we need to check with all queens
  # in current solution
  for i in range(0, proposed_row):
    old_row = i
    old_col = solution[i]

    diagonal_offset = proposed_row - old_row
    if (old_col == proposed_col or
      old_col == proposed_col - diagonal_offset or
        old_col == proposed_col + diagonal_offset):
      return False

  return True

def solve_n_queens(n):
  results = []
  solution = [-1] * n
  sol_stack = []

  row = 0
  col = 0

  while row < n:
    # For the current state of the solution, check if a queen can be placed in any
    # column of this row
    while col < n:
      if is_valid_move(row, col, solution):
        # If this is a safe position for a queen (a valid move), save 
        # it to the current solution on the stack...
        sol_stack.append(col)
        solution[row] = col
        row = row + 1
        col = 0
        # ... and move on to checking the next row (breaking out of the inner loop)
        break
      col = col + 1

    # If we have checked all the columns
    if col == n:
      # If we are working on a solution
      if sol_stack:
        # Backtracking, as current row does not offer a safe spot given the previous move
        # So, get set up to check the previous row with the next column
        col = sol_stack[-1] + 1
        sol_stack.pop()
        row = row - 1
      else:
        # If we have backtracked all the way and found this to be a dead-end,
        # break out of the inner loop
        break  # no more solutions exist
      
    # If we have found a safe spot for a queen in each of the rows
    if row == n:
      # add the solution into results
      results.append(solution)

      # backtrack to find the next solution
      row = row - 1
      col = sol_stack[-1] + 1
      sol_stack.pop()

  return len(results)

## Word Search 
'''
Given an m×n 2-D grid of characters, we have to find a specific word in the grid by combining the adjacent characters. Assume that only up, down, right, and left neighbors are considered adjacent.
Flow:
Start traversing the grid.
Call depth-first-search to search for the next character of the search word in four possible directions for each cell of the grid.
If a valid character is found, then call the depth-first-search function again for this cell.
Keep traversing the cells until the grid is empty or the valid string is found.
O(n*3^l) - O(l)
'''
# Function to search a specific word in the grid
def word_search(grid, word):
    n = len(grid)
    if n < 1:
        return False
    m = len(grid[0])
    if m < 1:
        return False
    for row in range(n):
        for col in range(m):
            if depth_first_search(row, col, word, grid):
                return True
    return False

# Apply backtracking on every element to search the required word
def depth_first_search(row, col, word, grid):
    if len(word) == 0:
        return True

    if row < 0 or row == len(grid) or col < 0 or col == len(grid[0]) \
            or grid[row][col].lower() != word[0].lower():
        return False

    result = False
    grid[row][col] = '*'

    for rowOffset, colOffset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        result = depth_first_search(row + rowOffset, col + colOffset,
                                    word[1:], grid)
        if result:
            break

    grid[row][col] = word[0]

    return result

## House Robber III
'''
A thief has discovered a new neighborhood to target, where the houses can be represented as nodes in a binary tree. The money in the house is the data of the respective node. The thief can enter the neighborhood from a house represented as root of the binary tree. Each house has only one parent house. The thief knows that if he robs two houses that are directly connected, the police will be notified. The thief wants to know the maximum amount of money he can steal from the houses without getting caught by the police. The thief needs your help determining the maximum amount of money he can rob without alerting the police.
Flow:
- If the tree is empty, return 0.
- Recursively calculate the maximum amount of money that can be robbed from the left and right subtrees of the root node.
- Recursively compute the maximum amount of money that can be robbed with the parent node included and excluded both.
- Return the maximum from both the amounts computed.
Naive approach: calculating the sum of every possible valid combination of houses that can be robbed. 2^n
Optimized solution: O(n) - O(n)
'''
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

        # below data members used only for some of the problems
        self.next = None
        self.parent = None
        self.count = 0

class BinaryTree:
    # constructor
    def __init__(self, *args):
        if len(args) < 1:
            self.root = None
        elif isinstance(args[0], int):
            self.root = BinaryTreeNode(args[0])
        else:
            self.root = None
            self.insertList(args[0])

    # function to create the binary tree given a list of integers
    def insertList(self, inputList):
        if not inputList:
            self.root = None
        else:
            self.root = BinaryTreeNode(inputList[0])
            q = deque([self.root])
            i = 1
            while i < len(inputList):
                currentNode = q.popleft()
                if inputList[i] != None:
                    currentNode.left = BinaryTreeNode(inputList[i])
                    q.append(currentNode.left)
                i += 1
                if i < len(inputList) and inputList[i] != None:
                    currentNode.right = BinaryTreeNode(inputList[i])
                    q.append(currentNode.right)
                i += 1

    # function to find a node given the value stored in the node
    def find(self, value):
        q = deque([self.root])
        while q:
            currentNode = q.popleft()
            if currentNode:
                if currentNode.data == value:
                    return currentNode
                q.append(currentNode.left)
                q.append(currentNode.right)
            if all(val == None for val in q):
                break
        return None

    # function to return level order traversal of the binary tree
    def level_order_traversal(self):
        if not self.root:
            return []
        result = []
        q = deque([self.root])
        while q:
            currentNode = q.popleft()
            if currentNode:
                result.append(currentNode.data)
                q.append(currentNode.left)
                q.append(currentNode.right)
            else:
                result.append(None)
            if all(val == None for val in q):
                break
        return result
    
def rob(root):
   # Returns maximum value from the pair: [includeRoot, excludeRoot]
   return max(heist(root))
   
def heist(root):
    # Empty tree case
    if root == None: 
        return [0, 0]

    # Recursively calculating the maximum amount that can be robbed from the left subtree of the root   
    left_subtree = heist(root.left) 
    # Recursively calculating the maximum amount that can be robbed from the right subtree of the root   
    right_subtree  = heist(root.right)
    # includeRoot contains the maximum amount of money that can be robbed with the parent node included
    includeRoot = root.data + left_subtree[1] + right_subtree[1] 
    # excludeRoot contains the maximum amount of money that can be robbed with the parent node excluded 
    excludeRoot = max(left_subtree) + max(right_subtree)

    return [includeRoot, excludeRoot] 

## Restore IP Addresses
'''
Given a string s containing digits, return a list of all possible valid IP addresses that can be obtained from the string. A valid IP address is made up of four numbers separated by dots ., for example 255.255.255.123. Each number falls between 0 and 255 (including 0 and 255), and none of them can have leading zeros.
Flow:
- Initially place all three dots each after 1 digit, e.g., 2.5.5.25511135
- Recursively add the next digit from right to the last segment of the IP address, e.g., 2.5.52.5511135
- If the number in any segment exceed 255, move this digit to the second segment.
- Make a condition that checks whether each segment lies within the range 0<=x<=255
- Once all dots are placed and each segment is valid, return the IP address.
Naive approach: The brute-force approach would be to check all possible positions of the dots. To place these dots, initially, we’ve 11 places, then 10 places for the second dot, 9 places for the third dot, and so on. So, in the worst case we would need to perform 11×10×9=990 validations.
Optimized approach: O(1) - O(1)
'''
def valid(segment):
    segment_length = len(segment)  # storing the length of each segment
    if segment_length > 3:  # each segment's length should be less than 3
        return False

    # Check if the current segment is valid
    # for either one of following conditions:
    # 1. Check if the current segment is less or equal to 255.
    # 2. Check if the length of segment is 1. The first character of segment
    #    can be `0` only if the length of segment is 1.
    return int(segment) <= 255 if segment[0] != '0' else len(segment) == 1


# this function will append the current list of segments to the list of result.
def update_segment(s, curr_pos, segments, result):
    segment = s[curr_pos + 1:len(s)]

    if valid(segment):  # if the segment is acceptable
        segments.append(segment)  # add it to the list of segments
        result.append('.'.join(segments))
        segments.pop()  # remove the top segment


def backtrack(s, prev_pos, dots, segments, result):
    # prev_pos : the position of the previously placed dot
    # dots : number of dots to place

    size = len(s)

    # The current dot curr_pos could be placed in
    # a range from prev_pos + 1 to prev_pos + 4.
    # The dot couldn't be placed after the last character in the string.
    for curr_pos in range(prev_pos + 1, min(size - 1, prev_pos + 4)):
        segment = s[prev_pos + 1:curr_pos + 1]
        if valid(segment):
            segments.append(segment)

            # if all 3 dots are placed add the solution to result
            if dots - 1 == 0:
                update_segment(s, curr_pos, segments, result)
            else:
                # continue to place dots
                backtrack(s, curr_pos, dots - 1, segments, result)

            segments.pop()  # remove the last placed dot


def restore_ip_addresses(s):

    # creating empty lists for storing valid IP addresses,
    # and each segment of IP
    result, segments = [], []
    backtrack(s, -1, 3, segments, result)
    return result

### *** Dynamic Programming
'''
There are multiple ways to solve dynamic programming problems: top-down, bottom-up, and optimized bottom-up approaches.
The bottom-up approach is optimized using the 1-D array instead of using the 2-D array.
'''
## 0/1 Knapsack
'''
Suppose you have the list of weights and corresponding values for n items. You have a knapsack that can carry a specific amount of weight at a time called capacity.
You need to find the maximum profit of items using the sum of values of the items you can carry in a knapsack. The sum of the weights of the items should be less than or equal to the knapsack’s capacity.
If any combination can’t make the given knapsack capacity of weights, then return 0.
Flow:
- Initialize the profits array of size capacity to save the profit for each capacity.
- Initialize a loop that will range from 0 to the length of values array.
- Initialize an inner loop that will range from capacity to 0.
- Check if weights[i] < capacity. If true, calculate the profit and save it in a profit array.
- Return profit[capacity].
Naive approach: ry all combinations to calculate the profit of the given values. And then choose one maximum profit from all the combinations with the weight that doesn’t exceed the given capacity. O(2^n) - O(n)
Optimized approach: O(nw) - O(w)
'''
def find_max_knapsack_profit(capacity, weights, values):
    # Store values length to use it later in the code
    values_length = len(values)
    # Check if the constraints are fulfilled for the given problem
    # Check if the given capacity is no smaller than or equal to zero
    # Check if the length of values is not equal to zero, if zero we will
    # return 0
    # Check if the length of weights is not equal to the length of the values,
    # if false we will return 0
    if capacity <= 0 or values_length == 0 or len(weights) != values_length:
        return 0
    # Initialize array named profits of size (capacity + 1) and
    # fill the array with 0    
    profits = [0] * (capacity + 1)
    # Iterate in values and weights list using i as an iterator where
    # values and weights list have same lengths
    for i in range(values_length):
        # Find the profit for each capacity starting from Cn to C0
        for c in range(capacity, -1, -1):
            # Check if the weight[i] is smaller than or equal to capacity
            # in range Cn - C0
            if weights[i] <= c:
                # Saving the profit for printing purposes
                init_profit = profits[c]
                # Calculate the new profit using the previous profit and
                # values[i]
                new_profit = profits[c - weights[i]] + values[i]
                # Set profits[c] value equal to the maximum of profits[c]
                # and new calculated profit
                profits[c] = max(profits[c], new_profit)
    return profits[capacity]

## Coin Change
'''
You're given an integer total and a list of integers called coins. The variable coins hold a list of coin denominations, and total is the total amount of money.
You have to find the minimum number of coins that can make up the total amount by using any combination of the coins. If the amount can't be made up, return -1. If the total amount is 0, return 0.
Flow:
- Initialize a counter array that contains elements equal to our total. Furthermore, initialize a variable to store the minimum number of coins needed. The minimum variable can be initialized to infinity at the start of each path.
- Traverse the coins array, and for each element, check the base cases. If the remaining sum is equal to 0, return 0. If it is less than 0, return -1 and if it is greater than 0, return the target value stored at the -ith index of the counter array. Store this value into a separate variable called result.
- Increment the value in result variable by one and add it to the minimum variable. Repeat this process until the (rem-1)^{th} index of the counter is not infinity.
Naive approach:  generates all possible combinations of given denominations such that in each combination, the sum of coins is equal to total. From these combinations, choose the one with the minimum number of coins and return the minimum number required. If the sum of any combinations is not equal to total then print -1. O(total^n) - O(n)
Optimized solution: O(n*m) - O(n)
'''
def calculate_minimum_coins(coins, rem, counter):  
    if rem < 0: 
        return -1
    if rem == 0:
        return 0
    if counter[rem - 1] != float('inf'):
        return counter[rem - 1]
    minimum = float('inf')

    for s in coins: 
        result = calculate_minimum_coins(coins, rem - s, counter)
        if result >= 0 and result < minimum:
            minimum = 1 + result

    counter[rem - 1] =  minimum if minimum !=  float('inf') else  -1 
    return counter[rem - 1]

def coin_change(coins, total): 
    if total < 1:
        return 0
    return calculate_minimum_coins(coins, total, [float('inf')] * total)

## N-th Tribonacci Number
'''
Given a number n, calculate the corresponding Tribonacci number. 
T0 = 0 , Y1 = 1 , T2 = 1 and T n+3 ​ =T n ​ +T n+1 ​ +T n+2 ​ , for n>=0
Flow:
- Initialize the first three numbers as 0, 1, and 1 respectively.
- If n is less than 3, then the result will be determined by the base case.
- Else continue computing the third and next numbers by adding the previous three numbers. Update them until the required number is obtained.
Naive approach: We’ll first initialize three numbers as 0, 1, and 1. If our required Tribonacci number is 1 or 2, we’ll return 1 in the base case. Otherwise, we’ll call the function recursively three times to compute the next number’s sum, as the tribonacci number is the sum of the previous three numbers. We’ll repeat this process until we hit the base case. O(3^n) - O(n)
Optimized approach: O(n) - O(1)
'''
def find_tribonacci(n):
    # if n is less than 3, then it's the base case
    if n < 3:
        return 1 if n else 0

    # initializing first three numbers
    first_num, second_num, third_num = 0, 1, 1
    # loop proceeds for n-2 times
    for _ in range(n - 2):
        # compute the sum of the last three Tribonacci numbers and update
        # first_num, second_num and third_num for the next iteration
        first_num, second_num, third_num = second_num, third_num, \
          first_num + second_num + third_num
    return third_num

## Partition Equal Subset Sum
'''
Given a non-empty array of positive integers, determine if the array can be divided into two subsets so that the sum of both the subsets is equal.
Flow:
- Create a matrix of size n∗s, where n is the size of the array and s is the sum of the array.
- Place TRUE in the first row of the matrix and FALSE in the first column of the matrix except the [0][0] location.
- Fill up the matrix in a bottom up approach.
- The last index of the matrix denotes whether the array can be partitioned or not.
Naive approach: First, we calculate the sum of the array. If the sum of the array is odd, there can’t be two subsets with the equal sum, so we return FALSE. If the sum is even, we calculate sum/2 and find a subset of the array with a sum equal to sum/2. This solution tests two possibilities, whether to include or exclude it. O(2^n)
Optimized approach: O(n*s)
'''
def can_partition_array(nums):
    
    # calculate sum of array
    t_sum = sum(nums)
    # if total sum is odd, it cannot be partitioned into equal sum subsets
    if t_sum % 2 != 0:
        return False
    
    s_sum = t_sum//2
    # Create a 2d matrix and fill all entries with False
    global matrix
    matrix = [[0 for i in range(len(nums)+1)] for j in range(s_sum+1)]
    # initialize the first row as true (1)
    # because each array has a subset whose sum is zero
    for i in range(0, len(nums) + 1):
        matrix[0][i] = 1

    # initialize the first column as false (0), except for [0][0] location
    # because an empty array has no subset whose sum is greater than zero
    for i in range(1, s_sum + 1):
        matrix[i][0] = 0
    # Fill up the 2d matrix in a bottom-up manner
    for i in range(1, s_sum+1):
        for j in range(1, len(nums)+1):
            if nums[j - 1] > i:
                matrix[i][j] = matrix[i][j - 1]
            else:
                matrix[i][j] = matrix[i - nums[j - 1]
                                      ][j - 1] or matrix[i][j - 1]

    # Print 2d array in matrix format
    print('\n'.join([''.join(['{:4}'.format(item)
              for item in row]) for row in matrix]))
              
    # Return the last index of the matrix, which is our answer
    return bool(matrix[s_sum][len(nums)])

## Word Break II
'''
Given a string s and a dictionary of strings word_dict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.
Flow:
- Identify all valid words that are a prefix of the input word.
- Split the word on all prefixes and the corresponding suffixes.
- For each valid word prefix, solve a sub-problem on the suffix
- If the suffix can be split into valid words, store the result in the output dictionary with each valid suffix as key and the list of words that compose the prefix as the value.
- Return the value from the dictionary that corresponds to the query string as the key.
Naive approach: use a basic recursive strategy in which we take each prefix of the string and compare it to the word in a dictionary. If it matches, we take the string’s suffix and repeat the process.
Optimized approach: O(nk + 2^n) - O((n*2^n) + k)
'''
def word_break(s, word_dict):
    return helper(s, word_dict, {}) #Calling the helper function
    
def helper(s, dictionary, result): #Helper Function that breaks down the string into words from subs
    if not s: #If s is empty string
        return []
    
    if s in result:
        return result[s]
    
    res = []
    for word in dictionary: #Verifying if s can be broken down further
        if not s.startswith(word):
            continue
        if len(word) == len(s):
            res.append(s)
        else:
            result_of_the_rest = helper(s[len(word):], dictionary, result)
            for item in result_of_the_rest:
                item = word + ' ' + item
                res.append(item)
    result[s] = res
    return res

### *** Cyclic Sort
'''
Cyclic sort is an in-place comparison sort algorithm. It is based on the idea that the array can be divided into cycles, and each element is rotated to create a sorted array. This pattern deals with problems involving arrays that contain numbers in a given range. Mostly, cyclic sort problems involve unique unsorted elements. This pattern is mostly used where we need to find the missing number or duplicate number in a given range.
'''
## Missing Number
'''
Given an array, nums, containing n distinct numbers in the range [0,n], return the only number in the range that is missing from the array.
Flow: 
- Start the list traversal from the first element.
- If the list element isn’t equal to its index, swap it with the number on the correct index.
- Else, if the element is at the correct index or greater than the length of the array, skip it and move one step forward.
- Once you’ve iterated over the entire array, compare each number with its index.
- The first occurrence of an index that’s not equal to its list element is the missing number.
Naive approach: A naive approach would be to first sort the array using quick sort and then traverse the array with two adjacent pointers. Since the integers are sorted, the difference between two adjacent array elements should be 1 if there is no missing integer. We can start by having the first pointer at index 0 and the second pointer at index 1, moving both 1 step forward each time. If the difference is greater than 1, our missing value would be the value of the first pointer + 1. O(nlogn) + O(n) - O(n)
Optimized approach: O(n) - O(1)
'''
def find_missing_number(nums):
  len_nums = len(nums)
  index = 0
  ind = 0

  while index < len_nums:
    ind+=1
    value = nums[index]

    if value < len_nums and value != nums[value]:
      nums[index], nums[value] = nums[value], nums[index]

    elif value >= len_nums:
      index+=1

    else:
      index += 1

  for x in range(len_nums):
    if x != nums[x]:
      return x
  return len_nums

## First Missing Positive
'''
Given an unsorted integer array, nums, return the smallest missing positive integer. Create an algorithm that runs with an O(n) time complexity and utilizes a constant amount of space.
Flow: 
- Traverse the array and make sure the elements present at the index are one less than their respective values if possible.
- Skip any negative numbers or the numbers that exceed that the length of our array.
- Traverse the array and check if the indices hold the value that is one more than the index itself.
- If no index contains any value other than one added to itself, add one to the largest number present in the array and return that as the output.
Naive approach: 
- The brute force approach would be to search for every positive integer in the array, starting from 1 and incrementing the number to be searched until we find the first missing positive number. O(n^2)
- A solution that everyone thinks about is sorting the array and then searching for the smallest positive number. O(nlogn)
- The final approach we would discuss is storing all the numbers in a hash table and returning the first positive value we don't encounter. 
Optimized approach: O(n) - O(1)
'''
def smallest_missing_positive_integer(nums):
  for i in range(len(nums)): #Traversing the array from the first index to the last index
    correct_spot = nums[i] - 1 #Determining what position the specific element should be at
    while 1 <= nums[i] <= len(nums) and nums[i] != nums[correct_spot]: 
      # The first condition verifies if there is a position
      # at which this specific number can be stored and the
      # second condition checks if the number is indeed at
      # the wrong position right now
      nums [i], nums[correct_spot] = nums[correct_spot], nums[i] # Swapping the number to it's correct position
      correct_spot = nums[i] - 1 # Making sure the new swapped number to the current index is verified in the
                                 # next iteration

  for i in range(len(nums)):
     if i + 1 != nums[i]:
       return i + 1
  return len(nums) + 1

## Find The Duplicate Number
'''
Given an array of integers, find a duplicate number such that the array contains n+1 integers in the range [1,n] inclusive.
There is only one repeated number in an array. You need to find that repeated number.
Flow:
- Traverse in an array using two pointers.
- Move the first pointer one time and the second pointer two times.
- Move until the pointers don’t meet.
- Move the first pointer from the start and the second pointer from the intersection point until they don’t meet.
Naive approach:The naive approach to this problem is sorting the given array. The duplicate numbers can be anywhere in an unsorted array, but they’ll be next to each other in a sorted array. After sorting the array, we can compare each index with index+1 to find any duplicate number. O(nlogn) - O(nlogn) or O(n)
Optimized approach: O(n) - O(1)
'''
def find_duplicate(nums):
    # Initialize the fast and slow pointers and make them point the first
    # element of the array
    fast = slow = nums[0]
    # PART #1
    # Traverse in array until the intersection point is not found
    while True:
        # Move the slow pointer using the nums[slow] flow
        slow = nums[slow]
        # Move the fast pointer two times fast as the slow pointer using the 
        # nums[nums[fast]] flow 
        fast = nums[nums[fast]]
        # Break of the slow pointer becomes equal to the fast pointer, i.e., 
        # if the intersection is found
        if slow == fast:
            break
    # PART #2
    # Make the slow pointer point the starting position of an array again, i.e.,
    # start the slow pointer from starting position
    slow = nums[0]
    # Traverse in an array until the slow pointer does not become equal to the
    # fast pointer
    while slow != fast:
        # Move the slow pointer using the nums[slow] flow
        slow = nums[slow]
        # Move the fast pointer slower than before, i.e., move the fast pointer
        # using the nums[fast] flow
        fast = nums[fast]
    # Return the fast pointer as it points the duplicate number of the array
    return fast

## Find the Corrupt Pair
'''
Given a non-empty unsorted array taken from a range of 1-n. Due to some data error, one of the numbers is duplicated, which results in another number missing. Create a function that returns the corrupt pair (missing, duplicated).
Flow:
- Apply cyclic sort on the array.
- After sorting, traverse the array and find the number which isn’t at its correct position. This will be the duplicate number.
- Add +1 to the index of the duplicate number. This will be our missing number.
- Return the pair containing the missing and duplicated number.
O(n) - O(1)
'''
def find_corrupt_pair(nums):
    # Initialize missing and duplicated
    missing = None
    duplicated = None

    # Function for swapping
    def swap(arr, first, second):
        arr[first], arr[second] = arr[second], arr[first]
    # Apply cyclic sort on the array

    i = 0
    # Traversing the whole array
    while i < len(nums):
        # Determining what position the specific element should be at
        correct = nums[i] - 1
        # Check if the number is at wrong position
        if nums[i] != nums[correct]:
            # Swapping the number to it's correct position
            swap(nums, i, correct)
        else:
            i += 1

    # Finding the corrupt pair(missing, duplicated)

    for j in range(len(nums)):
        if nums[j] != j + 1:
            duplicated = nums[j]
            missing = j + 1
    return missing, duplicated

## Find the First K Missing Positive Numbers
'''
For a given unsorted array, find the first k missing positive numbers in that array.
Flow:
- Sort the given input array by swapping the values into their correct positions.
- If the element is at its correct index or greater than the array’s length, skip it and move to the next element.
- Continue to check all the elements in this manner until the array is fully sorted.
- Next, compare each element with its index. Loop through the array and find all possible values that do not lie in the given array.
'''
### *** Topological Sort
'''
The topological sort pattern is used to find valid orderings of elements that have dependencies on, or priority over each other. Scheduling and grouping problems that have prerequisites or dependencies generally fall under this pattern.
'''
## Compilation Order
'''
There are a total of n classes labeled with the English alphabet (A, B, C, and so on). Some classes are dependent on other classes for compilation. For example, if class B extends class A, then B has a dependency on A. Therefore, A must be compiled before B.
Given a list of the dependency pairs, find the order in which the classes should be compiled.
Flow:
- Build the graph from the input using adjacency lists.
- Store the in-degree of each vertex in a hash map.
- Pop from the queue and store the node in a list, let’s call it sorted order.
- Decrement the in-degrees of the node’s children by 1. If the in-degree of a node becomes 0, add it to the source queue.
- Repeat until all vertices have been visited. Return the sorted order list.
Naive approach: generate all possible compilation orders for the given classes and then select the ones that satisfy the dependencies. O(n!) - O(1)
Optimized approach: O(V + E) - O(V)
'''
def find_compilation_order(dependencies):
    sorted_order = []
    graph = {}
    inDegree = {}
    for x in dependencies:
        parent, child = x[1], x[0]
        graph[parent], graph[child] = [], []
        inDegree[parent], inDegree[child] = 0, 0
    if len(graph) <= 0:
        return sorted_order


    for dependency in dependencies:
        parent, child = dependency[1], dependency[0]
        graph[parent].append(child)  
        inDegree[child] += 1  

    sources = deque()
    for key in inDegree:
        if inDegree[key] == 0:
            sources.append(key)

    while sources:
        vertex = sources.popleft()
        sorted_order.append(vertex)
        for child in graph[vertex]: 
            inDegree[child] -= 1
            if inDegree[child] == 0:
                sources.append(child)

    if len(sorted_order) != len(graph):
        return []
    return sorted_order

## Alien Dictionary
'''
In this challenge, you are given a list of words written in an alien language, where the words are sorted lexicographically by the rules of this language. Surprisingly, the aliens also use English lowercase letters, but possibly in a different order.
Given a list of words written in the alien language, you have to return a string of unique letters sorted in the lexicographical order of the alien language as derived from the list of words.
If there’s no solution, that is, no valid lexicographical ordering, you can return an empty string.
Flow: 
- Build the graph from the input using adjacency lists.
- Remove the sources, i.e. vertices with indegree=0 from the graph and add them to a results array.
- Decrement the indegree of the sources children by 1.
- Repeat until all nodes are visited.
Naive approach: The naive approach is to generate all possible orders of alphabets in the alien language and then iterate over them, character by character, to select the ones that satisfy the dictionary dependencies. O(u!) - O(1) - u is the number of unique alphabets in the alien language.
Optimized approach: O(c) - O(1)
'''
from collections import defaultdict, Counter, deque

def alien_order(words):
    adj_list = defaultdict(set)
    counts = Counter({c: 0 for word in words for c in word})
    outer = 0
    for word1, word2 in zip(words, words[1:]):
        outer += 1
        inner = 0
        for c, d in zip(word1, word2):
            inner += 1
            if c != d:
                if d not in adj_list[c]:
                    adj_list[c].add(d)
                    counts[d] += 1
                break

        else:  
            if len(word2) < len(word1):
                return ""

    
    result = []
    sources_queue = deque([c for c in counts if counts[c] == 0])
    while sources_queue:
        c = sources_queue.popleft()
        result.append(c)

        for d in adj_list[c]:
            counts[d] -= 1
            if counts[d] == 0:
                sources_queue.append(d)

    if len(result) < len(counts):
        return ""
    return "".join(result)

## Verifying an Alien Dictionary
'''
You’re given a list of words with lowercase English letters in a different order, written in an alien language. The order of the alphabet is some permutation of lowercase letters of the English language.
We have to return TRUE if the given list of words is sorted lexicographically in this alien language.
Flow:
- Store the ranking of each letter from the order string in the data structure.
- Iterate over the two adjacent words in the words list.
- If words[i + 1] ends before words[i], return FALSE.
- Else, if the characters in both of the words are different and words are in correct order, exit and move to the next two adjacent words.
- Else, return FALSE if the characters are different in both words and words are not in correct order.
- At the end of the loop, all of the adjacent words have been examined, which ensures that all of the words are sorted. Therefore, return TRUE.
Naive approach: The naive approach for this problem is to iterate over the order list and words simultaneously. Start from the first word, then compare the order with all other words present in the list. O(n^3)
Optimized approach: O(m) - O(1) - m is the total number of letters in the list of words.
'''
def verify_alien_dictionary(words, order):
    # If there is only one word to check, this is a trivial case with 
    # not enough input (minimum two words) to run the algorithm. 
    # So we return True
    if len(words) == 1:
        return True
    # Declare a hash map to store the characters of the words
    order_map = {}
    # Traverse order and store the rank of each letter in order_map
    for index, val in enumerate(order): 
        order_map[val] = index
    # Traverse in array words
    for i in range(len(words) - 1):
        # Traverse each character in a word
        for j in range(len(words[i])):
            # If all the letters have matched so far, but the current word
            # is longer than the next one, the two are not in order and
            # we return False
            if j >= len(words[i + 1]):
                return False
            # Check if the letters in the same position in the two words 
            # are different
            if words[i][j] != words[i + 1][j]:
                # Check if the rank of the letter in the current word is 
                # greater than the rank in the same position in the next word
                if order_map[words[i][j]] > order_map[words[i + 1][j]]:
                    return False
                # if we find the first different character and they are sorted,
                # then there's no need to check remaining letters
                break
    return True

## Course Schedule II
'''
Let’s assume that there are a total of n courses labeled from 0 to n−1. Some courses may have prerequisites. A list of prerequisites is specified such that if Prerequisitesi​ =a,b, you must take course b before course a.
Given the total number of courses n and a list of the prerequisite pairs, return the course order a student should take to finish all of the courses. If there are multiple valid orderings of courses, then the return any one of them.
Flow:
- Create a graph with a node for each course and edges representing the dependencies. Store the in-degrees of each node in a separate data structure.
- Pick a node with in-degree equal to zero and add it to the output list.
- Decrement the in-degree of the node picked in the previous step.
- Repeat for all nodes with in-degree equal to zero.
Naive approach: use nested loops to iterate over the prerequisites list, find the dependency of one course on another, and store them in a separate array. O(n^2) - O(n)
Optimized approach: 
- Make a graph by initializing the hash map with the vertices and their children.
- Keep the number of in-degrees of each of the vertex in the separate hash map.
- Find the source vertex.
- Add the source to the sorted list.
- Retrieve all the source children and add them to the queue.
- Decrement the in-degrees of the retrieved children.
- If the in-degree of the child vertex becomes equal to zero, add it to the sorted list.
- Repeat the process until the queue isn’t empty.
O(V + E) - O(V + E)
'''
def find_order(n, prerequisites):
    sorted_order = []
    # if n is smaller than or equal to zero we will return the empty array
    if n <= 0:
        return sorted_order

    # Store the count of incoming prerequisites in a hashmap
    in_degree = {i: 0 for i in range(n)}  
    # a. Initialize the graph
    graph = {i: [] for i in range(n)}  # adjacency list graph

    # b. Build the graph
    for prerequisite in prerequisites:
        parent, child = prerequisite[1], prerequisite[0]
        graph[parent].append(child)  # add the child to its parent's list
        in_degree[child] += 1  # increment child's in_degree

    # c. Find all sources i.e., all nodes with 0 in-degrees
    sources = deque()
    # traverse in in_degree using key
    for key in in_degree:
        # if in_degree[key] is 0 append the key in the deque sources
        if in_degree[key] == 0:
            sources.append(key)

    # d. For each source, add it to the sorted_order and subtract one from 
    # all of its children's in-degrees. If a child's in-degree becomes zero, 
    # add it to the sources queue
    while sources:
        # pop an element from the start of the sources and store 
        # the element in vertex
        vertex = sources.popleft()
        # append the vertex at the end of the sorted_order
        sorted_order.append(vertex)
        # traverse in graph[vertex] using child
        # get the node's children to decrement 
        # their in-degrees
        for child in graph[vertex]:  
            in_degree[child] -= 1
            # if in_degree[child] is 0 append the child in the deque sources
            if in_degree[child] == 0:
                sources.append(child)

    # topological sort is not possible as the graph has a cycle
    if len(sorted_order) != n:
        return []

    return sorted_order

## Course Schedule
'''
There are a total of num_courses courses you have to take. The courses are labeled from 0 to num_courses - 1. You are also given a prerequisites array, where prerequisites[i] = [a[i], b[i]] indicates that you must take course b[i] first if you want to take the course a[i]. For example, the pair [1,0] indicates that to take course 1, you have to first take course 0.
Return TRUE if all of the courses can be finished. Otherwise, return FALSE.
Flow:
- Initialize a graph containing the key as the parent and the value as its child’s vertices.
- Find all sources with 0 in-degrees.
- For each source, append it to a sorted list, retrieve all of its children, and decrement the in-degrees of the respective child by 1.
- Repeat the process until the source queue becomes empty.
O(V + E) - O(V + E)
'''
def can_finish(num_courses, prerequisites):
    sorted_order = []
    if num_courses <= 0:
        return True

    # a. Initialize the graph
    # count of incoming prerequisites
    inDegree = {i: 0 for i in range(num_courses)}
    graph = {i: [] for i in range(num_courses)}  # adjacency list graph

    # b. Build the graph
    for edge in prerequisites:
        parent, child = edge[1], edge[0]
        graph[parent].append(child)  # put the child into it's parent's list
        inDegree[child] += 1  # increment child's inDegree

    # c. Find all sources i.e., all num_courses with 0 in-degrees
    sources = deque()
    for key in inDegree:
        if inDegree[key] == 0:
            sources.append(key)

    # d. For each source, add it to the sorted_order and subtract one from
    # all of its children's in-degrees
    # if a child's in-degree becomes zero, add it to the sources queue
    while sources:
        course = sources.popleft()
        sorted_order.append(course)
        # get the node's children to decrement their in-degrees
        for child in graph[course]:
            inDegree[child] -= 1
            if inDegree[child] == 0:
                sources.append(child)

    # topological sort is not possible as the graph has a cycle
    if len(sorted_order) != num_courses:
        return False

    return True

## Find All Possible Recipes from Given Supplies
'''
A list of recipes a chef can prepare from the supplied items is given. Ingredients required to prepare a recipe are mentioned in the ingredients list. The ith recipe has the name recipes i, and you can create it if you have all the needed ingredients from the ingredients i list. A recipe may be listed as an ingredient in a different recipe. For example, the input may specify that custard is an ingredient in a trifle recipe or that trifle is an ingredient in a custard recipe.
Identify which recipes a chef can prepare from the given ingredients from the supplies list.
Flow:
- Calculate the count of the ingredients of each recipe. This is the count of the dependencies of each recipe.
- Start the topological sort with the list of supplies as the starting point.
- Use the topological sort to decrease the dependency count of each recipe.
- Scan through the list of recipes and add those to the result list whose dependency count is 0, that is, those for which all ingredients (whether as supplies, or as the results of other recipes) are available.
'''
### *** Stacks
'''
A stack is a linear data structure that follows a last in, first out (LIFO) order to perform operations on its elements. This means that whenever we obtain an element from the stack, it will always return the last inserted element.
Stacks in programming are used to store elements that are sequentially dependent on each other. When required, the elements can then be popped from the stack while maintaining a fixed order.
'''
## Basic Calculator
'''
Given a string containing an arithmetic expression, implement a basic calculator that evaluates the expression string. The expression string can contain integer numeric values and should be able to handle the “+” and “-” operators, as well as “()” parentheses.
Flow:
- Initialize three variables: number to store the integer form of the current number in the string (initially 0), sign value to store a multiplication value to change the sign (initially 1), and result to store the evaluated result of different operations (initially 0).
- Upon encountering a digit character, update the number variable by multiplying its existing value by 10 and adding the integer value of the digit character to it: number = number × 10 + digit
- Upon encountering a ‘(’ character, push the value of the result variable and then the sign value onto the stack. In addition, reset the value of the sign value, and result variable to 1, 0 respectively.
- Upon encountering a ‘+’ or ‘-’ character, change the sign value variable to 1 or -1 respectively. Then evaluate the expression on the left by multiplying the existing value of the result variable by the sign value variable and adding the number to this: result = number + (result × sign value)
- In addition, reset the value of the number variable to 0.
- Upon encountering a ‘)’ character, update the result variable to evaluate the expression within the parenthesis: result = number + (result × sign value)
- Then pop the sign value and stored digit from the stack and update the result variable again: result = (result × sign value) + digit
- In addition, reset the value of the number variable to 0.
O(n) - O(n)
'''
def calculator(expression):
    number = 0
    sign_value = 1
    result = 0
    operations_stack = []

    for c in expression:
        if c.isdigit():
            number = number * 10 + int(c)
        if c in "+-":
            result += number * sign_value
            sign_value = -1 if c == '-' else 1
            number = 0
        elif c == '(':
            operations_stack.append(result)
            operations_stack.append(sign_value)
            result = 0
            sign_value = 1

        elif c == ')':
            result += sign_value * number
            pop_sign_value = operations_stack.pop()
            result *= pop_sign_value

            second_value = operations_stack.pop()
            result += second_value
            number = 0
    
    return result + number * sign_value

## Remove All Adjacent Duplicates In String
'''
Given a string consisting of lowercase English letters, repeatedly remove adjacent duplicate letters, one pair at a time. Both members of a pair of adjacent duplicate letters need to be removed.
Flow:
- Initialize a stack to store string characters and their frequencies.
- Iterate over the string. If the current character and the character at the top of the stack are the same, increment its count. Otherwise, push it to the stack and set its count to 1.
- Check if the count on the top of the stack reaches 2. If yes, we’ve found the adjacent duplicates. hence, we pop from the stack.
- Once the traversal is complete, the characters stored in the stack are the required sequence of characters, with all adjacent duplicates removed.
Naive approach: generate a set of all possible adjacent duplicates in the English language and traverse over the string to see if these duplicates are present. We’ll use nested loops to check this condition. If yes, we’ll replace the adjacent duplicates in the input string with an empty character. O(n^2)
Optimized approach: O(n) - O(n)
'''
def remove_duplicates(to_clean_up):
    frequency_stack = []
    k = 2
    i = 0
    for char in to_clean_up:  
        if frequency_stack and frequency_stack[-1][0] == char:
            frequency_stack[-1][1] += 1

            if frequency_stack[-1][1] == k:
                frequency_stack.pop()
        else:
            frequency_stack.append([char, 1])
        i += 1

    result = ""
    for elem in frequency_stack:
        char = elem[0]
        count = elem[1]
        result = result + (char * count)
    return result

## Minimum Remove to Make Valid Parentheses
'''
Given a string with matched and unmatched parentheses, remove the minimum number of parentheses so that the resulting string is a valid parenthesization.
Flow:
- Traverse the string, while keeping track of the parenthesis alongside their indices in the stack.
- If matched parenthesis is found, remove it from the stack.
- Once the string has been traversed, we will be left only with the unmatched parentheses in the stack.
- Create a new string without including the characters at indices still present in the stack.
Naive approach: raverse the string twice. Each time we find a parenthesis in the string, we’ll traverse the string from the start to the end to find if this parenthesis is matched or not. O(n^2) - O(n)
Optimized approach: O(n) - O(n)
'''
def min_remove_parentheses(s):
    delimiters = []
    builder = s
    for i, val in enumerate(s):
        if len(delimiters) > 0 and delimiters[-1][0] == '(' and val == ')':
            # closing parenthesis found while top of delimiters stack
            # is an opening parenthesis, so pop the
            # opening parenthesis as it's part of a pair
            delimiters.pop()
        elif val == '(' or val == ')':
            # parenthesis found, push to delimiters stack for checking
            # against the rest of the string
            delimiters.append([val, i])

    # At this point, the delimiters stores the indices
    # that need to be removed from the input string
    while delimiters:
        # compile the result string, skipping the
        # indices that need to be removed from the input
        popped_val = delimiters.pop()
        index = popped_val[1]
        builder = builder[0:index] + builder[index+1:len(builder)]
    return builder

## Exclusive Execution Time of Functions
'''
We are given an integer number n , representing the number of functions running in a single-threaded CPU, and an execution log, which is essentially a list of strings. Each string has the format {function id}:{"start" | "end"}:{timestamp}, indicating that the function with function_id either started or stopped execution at time identified by the timestamp value. Each function has a unique ID between 0 and n−1 . Compute the exclusive time of the functions in the program.
Flow:
- Retrieve function ID, start/end and timestamp from the log string.
- If the string contains “start”, push the log details to the stack.
- Else if, the string contains “end”, pop from the stack and compute the function’s execution time.
- If the stack is not empty after the pop operation, subtract the execution time of the called function from the calling function.
- Store the execution time in a results array and return.
Naive approach: select each function and track the time it takes by subtracting the start time from the end time. However, since there are other functions that can have overlapping times, we would need to keep track of function preemptions. We would need to adjust the previously computed execution time of functions that are found to have been preempted. O(n^n) - O(n)
Optimized solution: O(mlogn) - O(mlogn)
'''
class Log:
    def __init__(self, content):
        content = content.replace(' ', '')
        content = content.split(":")
        self.id = int(content[0])
        self.is_start = content[1] == "start"
        self.time = int(content[2])

def exclusive_time(n, logs):
    logs_stack = []
    result = [0]*n

    for content in logs:
        # Extract the logs details from the content(string)
        logs = Log(content)
        if logs.is_start:
            # Push the logs details to the stack
            logs_stack.append(logs)
        else:
            # Pop the logs details from the stack
            top = logs_stack.pop()
            # Add the execution time of the current function in the actual result
            result[top.id] += (logs.time - top.time + 1)
            # If the stack is not empty, subtract the current child function execution time from the parent function
            if logs_stack:
                result[logs_stack[-1].id] -= (logs.time - top.time + 1)
    return result

## Flatten Nested List Iterator
'''
You’re given a nested list of integers. Each element is either an integer or a list whose elements may also be integers or other integer lists. Your task is to implement an iterator to flatten the nested list.
You will have to implement the Nested Iterator class. This class has the following functions:
Init (nested list): This initializes the iterator with the nested list.
Next (): This returns the next integer in the nested list.
Has Next (): This returns TRUE if there are still some integers in the nested list. Otherwise, it returns FALSE.
Flow:
- If the top element of the stack is an integer, return TRUE.
- If the top element of the stack is a list of integers, pop the list and push each element of the list into the stack in reverse order and return TRUE.
- If the stack is empty, return FALSE.
O(l/n) - O(l + n)
'''
class NestedIntegers:
    # Constructor initializes a single integer if a value has been passed
    # else it initializes an empty list
    def __init__(self, integer=None):
        if integer:
            self.integer = integer
        else:
            self.n_list = []
            self.integer = 0 

    # If this NestedIntegers holds a single integer rather 
    # than a nested list, returns TRUE, else, returns FALSE
    def is_integer(self):
        if self.integer:
            return True
        return False

    # Returns the single integer, if this NestedIntegers holds a single integer
    # Returns null if this NestedIntegers holds a nested list
    def get_integer(self):
        return self.integer

    #  Sets this NestedIntegers to hold a single integer.
    def set_integer(self, value):
        self.n_list = None
        self.integer = value

    # Sets this NestedIntegers to hold a nested list and adds a nested 
    # integer to it.
    def add(self, ni):
        if self.integer:
            self.n_list = [] 
            self.n_list.append(NestedIntegers(self.integer)) 
            self.integer = None
        self.n_list.append(ni) 

    # Returns the nested list, if this NestedIntegers holds a nested list 
    # Returns null if this NestedIntegers holds a single integer
    def get_list(self):
        return self.n_list
    
class NestedIterator:

    # NestedIterator constructor initializes the stack using the 
    # given nested_list list
    def __init__(self, nested_list):
        self.nested_list_stack = list(reversed(nested_list))

    # has_next() will return True if there are still some integers in the 
    # stack (that has nested_list elements) and, otherwise, will return False.
    def has_next(self):
        # Iterate in the stack while the stack is not empty
        while len(self.nested_list_stack) > 0: 
            # Save the top value of the stack
            top = self.nested_list_stack[-1]
            # Check if the top value is integer, if true return True, 
            # if not continue
            if top.is_integer():
                return True
            # If the top is not an integer, it must be the list of integers
            # Pop the list from the stack and save it in the top_list
            top_list = self.nested_list_stack.pop().get_list()
            # Save the length of the top_list in i and iterate in the list
            i = len(top_list) - 1
            while i >= 0:
                # Append the values of the nested list into the stack
                self.nested_list_stack.append(top_list[i])
                i -= 1
        return False

    # next will return the integer from the nested_list 
    def next(self): 
        # Check if there is still an integer in the stack
        if self.has_next():
            # If true pop and return the top of the stack
            return self.nested_list_stack.pop().get_integer()
        return None

## Valid Parentheses
'''
Given a string that may consist of opening and closing parentheses, your task is to check if the string contains valid parenthesization or not.
Flow:
- If the current character is an opening parenthesis, push it onto the stack.
- Else, if the current character is a closing parenthesis and it corresponds to the opening parenthesis on the top of the stack, then pop from the stack.
- After complete traversal, if the stack is empty then the parentheses are valid. If the stack is not empty, then the parentheses are not valid.
'''
### *** Tree Depth-first Search
'''
A tree is a graph that contains the following properties:
- It is undirected.
- It is acyclic (contains no cycles).
- It consists of a single connected component.
There are three main methods to solving a problem with the depth-first search pattern—Preorder, Inorder, and Postorder.
'''
## Flatten Binary Tree to Linked List
'''
Given the root of a binary tree, flatten the tree into a linked list using the same Tree class. The left child of the linked list is always NULL, and the right child points to the next node in the list. The nodes in the linked list should be in the same order as the preorder traversal of the given binary tree.
Flow:
- For every node, check whether it has a left child or not. If it does not have a left child, we move on to the right child.
- Otherwise, find the node on the rightmost branch of the left subtree that does not have a right child.
- Once we find this rightmost node, connect it with the right child of the current node. After connecting, set the right child of the current node to the left child of the current node.
- Finally, set the left child of the current node to NULL.
- Repeat the process until the given binary tree becomes flattened.
Naive approach: perform level order traversal of binary tree using the Queue. During level order traversal, keep track of the previous node. We’d make the current node the right child of the last node and the left of the last node NULL. After this traversal, the binary tree will be flattened as a linked list. O(n) space for queue.
Optimized approach: O(1) - O(1)
'''
class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

        # below data member is only used for printing
        self.printData = str(data)

        # below data members used only for some of the problems
        self.next = None
        self.parent = None
        self.count = 0

class BinaryTree:
    def __init__(self, *args):
        if len(args) < 1:
            self.root = None
        elif isinstance(args[0], int):
            self.root = BinaryTreeNode(args[0])
        else:
            self.root = None
            for x in args[0]:
                self.insert(x)

    # for BST insertion
    def insert(self, node_data):
        new_node = BinaryTreeNode(node_data)
        if not self.root:
            self.root = new_node
        else:
            parent = None
            temp_pointer = self.root
            while temp_pointer:
                parent = temp_pointer
                if node_data <= temp_pointer.data:
                    temp_pointer = temp_pointer.left
                else:
                    temp_pointer = temp_pointer.right
            if node_data <= parent.data:
                parent.left = new_node
            else:
                parent.right = new_node

    def find_in_bst_rec(self, node, node_data):
        if not node:
            return None
        if node.data == node_data:
            return node
        elif node.data > node_data:
            return self.find_in_bst_rec(node.left, node_data)
        else:
            return self.find_in_bst_rec(node.right, node_data)

    def find_in_bst(self, node_data):
        return self.find_in_bst_rec(self.root, node_data)

    def get_sub_tree_node_count(self, node):
        if not node:
            return 0
        else:
            return 1 + self.get_sub_tree_node_count(node.left) + self.get_sub_tree_node_count(node.right)

    def get_tree_deep_copy_rec(self, node):
        if node:
            new_node = BinaryTreeNode(node.data)
            new_node.left = self.get_tree_deep_copy_rec(node.left)
            new_node.right = self.get_tree_deep_copy_rec(node.right)
            return new_node
        else:
            return None

    def get_tree_deep_copy(self):
        if not self.root:
            return None
        else:
            tree_copy = BinaryTree()
            tree_copy.root = self.get_tree_deep_copy_rec(self.root)
            return tree_copy
        
def flatten_tree(root):
    if not root:
        return
    # Assign current to root
    current = root

    # Traversing the whole tree
    print("  Traversing the tree\n")
    while current:
        if current.left:
            print("  The current node has a left child\n")
            last = current.left

            # If the last node has right child
            while last.right:
                print("  The current node has a right child\n")
                last = last.right
            
            # If the last node does not have right child
            print("   The current node does not have a right child")
            print("   We'll merge it with the right subtree")
            last.right = current.right
            current.right = current.left
            current.left = None            
        if current.right:
            print("  Moving to the right child\n")
        current = current.right
    return root

## Diameter of Binary Tree
'''
Given a binary tree, you need to compute the length of the tree’s diameter. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
Flow:
- Start traversing the tree from the root node.
- For each node, calculate the height of the left and right subtree.
- For each node, update the diameter using the following formula: max(diameter, left height + right height).
- After traversing the whole tree, return the diameter value since it is the length of the tree’s diameter.
Naive approach: pick one node, find the distance to every other node from the selected node, and keep track of the maximum value. Repeat this process with all the nodes of the tree. After this process, we’ll have the maximum possible distance between the two nodes. That will be the diameter of the tree. O(n^2)
Optimized approach: O(n) - O(n)
'''
# function to recursively calculate the height of the tree
# and update the diameter
def diameter_helper(node, diameter_res):
    if node is None:
        return 0, diameter_res
    else:
        # Compute the height of each subtree
        lh, diameter_res = diameter_helper(node.left, diameter_res)
        rh, diameter_res = diameter_helper(node.right, diameter_res)

        # update the result with the max of the previous and current diameter value
        diameter_res = max(diameter_res, lh + rh)

        # Use the larger one
        return max(lh, rh) + 1, diameter_res
    
def diameter_of_binaryTree(root):
    # variable for diameter
    diameter_res = 0
    if not root:
        return 0
    # compute the height of the tree and the max diameter
    _, diameter_res = diameter_helper(root, diameter_res)
    # return the diameter
    return diameter_res

## Serialize and Deserialize Binary Tree
'''
Serialize a given binary tree to a file and deserialize it back to a tree. Make sure that the original and the deserialized trees are identical.
Serialize: Write the tree to a file.
Deserialize: Read from a file and reconstruct the tree in memory.
Serialize the tree into a list of integers, and then, deserialize it back from the list to a tree. For simplicity’s sake, there’s no need to write the list to the files.
Flow:
- Perform a depth-first traversal and serialize individual nodes to the stream.
- Also, serialize a marker to represent a NULL pointer that helps deserialize the tree.
- Deserialize the tree using preorder traversal.
- During deserialization, create a new node for every non-marker node using preorder traversal.
Naive approach:  store one of the traversals of the tree into a file when serializing a tree and read that traversal back to create a tree when deserializing. However, any one of the traversals is not unique. That is, two different trees can have the same in-order traversal. The same goes for pre-order or post-order traversal as well. 
For serialization, this approach will store both the inorder and preorder traversal and place a delimiter to separate them.
For deserialization, pick each node from the preorder traversal, make it root, and find its index in the inorder traversal. The nodes to the left of that index will be the part of the left subtree, and the nodes to the right of that index will be the part of the right subtree.
Optimized approach: O(n) - O(logn) for balanced tree/ O(n) for degenerate tree
'''
# Initializing our marker
MARKER = "M"
m = 1


def serialize_rec(node, stream):
    global m
    # Adding marker to stream if the node is None
    if node is None:
        stream.append(MARKER + str(m))
        m += 1
        return

    # Adding node to stream
    stream.append(node.data)

    # Doing a pre-order tree traversal for serialization
    serialize_rec(node.left, stream)
    serialize_rec(node.right, stream)

# Function to serialize tree into list of integers.
def serialize(root):
    stream = []
    serialize_rec(root, stream)
    return stream

def deserialize_helper(stream):
    # pop last element from list
    val = stream.pop()

    # Return None when a marker is encountered
    if type(val) is str and val[0] == MARKER:
        return None

    # Creating new Binary Tree Node from current value from stream
    node = BinaryTreeNode(val)

    # Doing a pre-order tree traversal for serialization
    node.left = deserialize_helper(stream)
    node.right = deserialize_helper(stream)

    # Return node if it exists
    return node

# Function to deserialize integer list into a binary tree.
def deserialize(stream):
    stream.reverse()
    node = deserialize_helper(stream)
    return node

## Invert Binary Tree
'''
Given the root node of a binary tree, convert the binary tree into its mirror image.
Flow:
- Perform post order traversal on the left child of the root node.
- Perform post order traversal on the right child of the root node
- Swap the left and right children of the root node.
O(n) - O(logn) for balanced tree/ O(n) for degenerate tree
'''
# global variables to support step-by-step printing
change = 0
master_root = None


# Function to mirror binary tree
def mirror_binary_tree(root):
    global change, master_root
    # Base case: end recursive call if current node is null
    if not root:
        return

    # We will do a post-order traversal of the binary tree.
    if root.left:
        mirror_binary_tree(root.left)

    if root.right:
        mirror_binary_tree(root.right)

    # Let's swap the left and right nodes at current level.
    root.left, root.right = root.right, root.left

    # Only to demonstrate step-by-step operation of the solution
    if master_root and (root.left or root.right):
        change += 1
        print("\n\tChange ", change, ", at node ", root.data, ":", sep="")
        #display_tree(master_root)

    return root

## Binary Tree Maximum Path Sum
'''
Given the root of a binary tree, return the maximum sum of any non-empty path.
A path in a binary tree is defined as follows:
- A sequence of nodes in which each pair of adjacent nodes must have an edge connecting them.
    * A node can only be included in a path once at most.
    * Including the root in the path is not compulsory.
You can calculate the path sum by adding up all node values in the path. To solve this problem, calculate the maximum path sum given the root of a binary tree so that there won’t be any greater path than it in the tree.
Flow: 
- Initialize a maximum sum to negative infinity.
- For a leaf node, determine its contribution equal to its value.
- Otherwise, determine a node’s contribution as its value plus the greater of the contributions of its left and right children.
- Update maximum sum if the above is greater than previous maximum sum.
O(n) - O(logn)/O(n)
'''
class MaxTreePathSum:
    max_sum = float('-inf')

    def __init__(self):
        self.max_sum = float('-inf')

    def max_contrib(self, root):
        if not root:
            return 0

        # sum of the left and right subtree
        max_left = self.max_contrib(root.left)
        max_right = self.max_contrib(root.right)

        left_subtree = 0
        right_subtree = 0

        # max sum on the left and right sub-trees of root
        if max_left > 0:
            left_subtree = max_left
        if max_right > 0:
            right_subtree = max_right

        # the value to start a new path where `root` is a highest root
        value_new_path = root.data + left_subtree + right_subtree

        # update max_sum if it's better to start a new path
        self.max_sum = max(self.max_sum, value_new_path)

        # for recursion :
        # return the max contribution if continue the same path
        return root.data + max(left_subtree, right_subtree)

    def max_path_sum(self, root):
        self.max_sum = float('-inf')
        self.max_contrib(root)
        return self.max_sum

### *** Tree Breadth-first Search
'''
Tree breadth-first search is an important tree traversal algorithm for finding a node in a tree that satisfies the given constraints. It starts searching at the root node and moves down level by level, exploring adjacent nodes at level k+1.
Essentially, it first visits nodes that are one edge away from the root, followed by nodes that are two edges away, and so on. This helps in efficiently finding neighbor nodes, particularly peer-to-peer networking systems.
'''
## Level Order Traversal of Binary Tree
'''
Given the root of a binary tree, display the values of its nodes while performing a level order traversal. Return the node values for all levels in a string separated by the character :. If the tree is empty, i.e., the number of nodes is 0, then return “None” as the output.
Flow:
- Declare two queues, current and next.
- Push the root node to the current queue and set the level to zero.
- Dequeue the first element from the current queue and push its children in the next queue.
- If the current queue is empty, increase the level number and swap the two queues.
- Repeat until the current queue is empty.
Naive approach: first calculate the height of the tree and then recursively traverse it, level by level. The printing function would be called for every level in the range [1,height]. O(n^2) - O(n)
Optimized approach: O(n) - O(n)
'''
# Using two queues
def level_order_traversal(root):
    #   We print None if the root is None
    if not root:
        print("None", end="")
    else:
        result = ""
        # Declaring an array of two queues
        queues = [deque(), deque()]
        # Initializing the current and next queues
        current_queue = queues[0]
        next_queue = queues[1]

        # Enqueuing the root node into the current queue and setting
        # level to zero
        current_queue.append(root)
        level_number = 0

        n = 0
        while current_queue:
            n += 1
            # Dequeuing and printing the first element of queue
            temp = current_queue.popleft()
            result += str(temp.data)
            # Adding dequeued node's children to the next queue
            if temp.left:
                next_queue.append(temp.left)
            if temp.right:
                next_queue.append(temp.right)

            # When the current queue is empty, we increase the level, print a new line
            # and swap the current and next queues
            if not current_queue:
                level_number += 1
                if next_queue:
                    result += " : "
                current_queue = queues[level_number % 2]
                next_queue = queues[(level_number + 1) % 2]
            else:
                result += ", "
        return result

#Using one queue
def level_order_traversal(root):
    result = ""
    # We print None if the root is None
    if not root:
        result += ("None")
    else:
        # Initializing the current queue
        current_queue = deque()

        # Initializing the dummy node
        dummy_node = BinaryTreeNode(0)

        current_queue.append(root)
        current_queue.append(dummy_node)

        # Printing nodes in level-order until the current queue remains
        # empty
        while current_queue:
            # Dequeuing and printing the first element of queue
            temp = current_queue.popleft()
            result += str(temp.data)

            # Adding dequeued node's children to the next queue
            if temp.left:
                current_queue.append(temp.left)

            if temp.right:
                current_queue.append(temp.right)

            # When the dummyNode comes next in line, we print a new line and dequeue
            # it from the queue
            if current_queue[0] == dummy_node:
                current_queue.popleft()

                # If the queue is still not empty we add back the dummy node
                if current_queue:
                    result += " : "
                    current_queue.append(dummy_node)
            else:
                result += ", "
    return result

## Binary Tree Zigzag Level Order Traversal
'''
Given a binary tree, return its zigzag level order traversal. The zigzag level order traversal corresponds to traversing nodes from left to right for one level, and then right to left for the next level, and so on, reversing direction after every level.
Flow:
- Add the root of the tree to the deque and initialize the order flag reverse to FALSE.
- Iterate over the deque as long as it’s not empty.
- For each dequed element, add its children to the deque from the front if the value of the reverse is TRUE. Otherwise, append them to the back of the deque.
- To deque elements, enque from the back if the value of the reverse is TRUE. Otherwise, enque them from the front and maitain the order of the nodes.
- For each level, append array of its nodes to the results array.
Naive approach: The naive approach would be to get the level order traversal of the binary tree using a queue. We’ll store the nodes at each level in an array and then reverse the contents of alternate levels in-place. O(n) - O(n)
Optimized approach: O(n) - O(n) .Even though the naive solution has the same time complexity, the optimized approach does the job in one BFS traversal, without the need for a two-dimensional list to store the nodes.
'''
def zigzag_level_order(root):
    if root is None:
        return []

    results = []
    reverse = False
    dq = deque([root])
    a = 1

    while(len(dq)):
        size = len(dq)
        results.insert(len(results), [])
        a = a + 1

        for i in range(size):
            if not reverse:
                node = dq.popleft()
                results[len(results) - 1].append(node.data)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
            else:
                node = dq.pop()
                results[len(results) - 1].append(node.data)
                if node.right:
                    dq.appendleft(node.right)
                if node.left:
                    dq.appendleft(node.left) 

        reverse = not reverse
    return results

## Populating Next Right Pointers in Each Node
'''
Given a binary tree, connect all nodes of the same hierarchical level. We need to connect them from left to right, so that the next pointer of each node points to the node on its immediate right. The next pointer of the right-most node at each level will be NULL.
For this problem, each node in the binary tree has one additional pointer (the next pointer) along with the left and right pointers.
Flow:
- Traverse the tree level by level, starting from the root node.
- If both children of the current node exist, first connect them with each other and then with the previous nodes at the same level.
- Else, if only the left child of the current node exists, connect it with the previous nodes at the same level.
- Else, if only the right child of the current node exists, connect it with the previous nodes at the same level.
- Set the next pointer of the right-most node to NULL and move to the next level.
- Stop traversing the tree when all nodes have been visited.
Naive approach: use a queue for the level order traversal of our binary tree. Since we also need to keep track of the node’s level, we can push a (node, level) pair to the queue. O(n) - O(n)
Optimized approach: O(n) - O(1)
'''
# Helper function to connect all children nodes at the next level
def connect_next_level(head):
    #   Declaring the necessary pointers
    current = head
    next_level_head = None
    prev = None

    while current:
        if current.left and current.right:
            # If both current node children are not null
            # then connect them with each other and previous
            # nodes in the same level.
            if not next_level_head:
                next_level_head = current.left

            current.left.next = current.right

            if prev:
                prev.next = current.left

            prev = current.right
        elif current.left:
            # If only the left child node is not null
            # then only connect it with previous same level nodes
            if not next_level_head:
                next_level_head = current.left

            if prev:
                prev.next = current.left

            prev = current.left
        elif current.right:
            # If only the right child node children is not null
            # then only connect it with previous same level nodes
            if not next_level_head:
                next_level_head = current.right

            if prev:
                prev.next = current.right

            prev = current.right

        # Update current pointer
        current = current.next

    # Pointing the last node (right-most node) of the next level
    # to None
    if prev:
        prev.next = None

    # Return the head node (left-most node) of the next level
    return next_level_head


# Function to populate same level pointers
def populate_next_pointers(node):
    if node:
        node.next = None

        while True:
            node = connect_next_level(node)
            if not node:
                break


# Function to find the given node and return its next node
def get_next_node(node, nodeData):
    # Performing Binary Search
    while node and nodeData != node.data:
        if nodeData < node.data:
            node = node.left
        else:
            node = node.right

    # If node is not found return -1 else return its next node
    if not node:
        non_existing_node = BinaryTreeNode(-1)
        return non_existing_node
    else:
        return node.next

## Vertical Order Traversal of a Binary Tree
'''
Find the vertical order traversal of a binary tree when the root of the binary tree is given. In other words, return the values of the nodes from top to bottom in each column, column by column from left to right. If there is more than one node in the same column and row, return the values from left to right.
Flow:
- Traverse the tree, level by level, starting from the root node.
- Push the nodes to a queue along with their column index.
- If a node has children, assign column index current−1 to the left child and current+1 to the right child.
- Keep track of the maximum and minimum column indices, and populate a hash map with (index,node) pairs.
- Return the node values for each column index, from minimum to maximum.
Naive approach: traversing the tree to get the maximum and minimum horizontal distance of the nodes from the root. Once we have these numbers, we can traverse over the tree again for each distance in the range [maximum, minimum] and return the nodes respectively. O(n^2) - O(n)
Optimized approach: O(n) - O(n)
'''
def vertical_order(root):
    if root is None:
        return []

    node_list = defaultdict(list)
    min_column = 0
    max_index = 0

    # push root into the queue
    queue = deque([(root, 0)])

    # traverse over the nodes in the queue
    while queue:
        node, column = queue.popleft()

        if node is not None:
            temp = node_list[column]
            temp.append(node.data)
            node_list[column] = temp

            # get min and max column numbers for the tree
            min_column = min(min_column, column)
            max_index = max(max_index, column)

            # add current node's left and right child in the queue
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))

    return [node_list[x] for x in range(min_column, max_index + 1)]

### *** Trie
'''
Trie is a tree data structure used for storing and locating keys from a set. The keys are usually strings that are stored character by character—each node of a trie corresponds to a single character rather than the entire key.
The order of characters in a string is represented by edges between the adjacent nodes. For example, in the string “are”, there will be an edge from node a to node r to node e. That is, node a will be the parent of node r, and node r will be the parent of node e.
'''
## Implement Trie
'''
Trie is a tree-like data structure used to store strings. The tries are also called prefix trees because they provide very efficient prefix matching operations. Implement a trie data structure with three functions that perform the following tasks:
- Insert a string.
- Search a string.
- Search for a given prefix in a string.
Flow: 
- Begin from the root node and iterate over the string one character at a time.
- At each node, check if the character present in the trie.
- If it’s not present, a new node is initialized.
- Else, if it’s present, move to the next node.
- For the last character of the word, set a boolean variable to TRUE for the corresponding node.
'''
class TrieNode():
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie():
    def __init__(self):
        self.root = TrieNode()

    # inserting string in trie
    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children.get(c)
        node.is_word = True  

    # searching for a string
    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children.get(c)
        return node.is_word

    def search_prefix(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children.get(c)
        return True

## Search Suggestions System
'''
Given an array of strings called products and a word to search, design a system that, when each character of the searched word is typed, suggests at most three product names from products. Suggested products should share a common prefix with the searched word. If more than three products exist with a common prefix, return the three product names that appear first in lexicographical order.
Return the suggested products, which will be a list of lists after each character of searched word is typed.
Flow:
- Insert all product names in a trie, creating a node for each new character in a word.
- As each new letter of the searched word is received, retrieve all words in the trie whose initial characters match. For example, if “gam” has been typed in, it may match “game”, “games”, “gamify” and “gamma”.
- If there are more than three matched strings, return the ones that appear first in the lexicographic order. In our example, we would return “game”, “games” and “gamify”.
Naive approach: A naive approach would be to first sort the given product data in lexicographic order to help with retrieving three matching products. Next we’ll make a substring for every character of the word to be searched. We’ll search in the list of product names to check whether the current substring exists or not. If it exists, we’ll store the results (containing matching product names) for the current substring. We’ll repeat this process until we have traversed the whole list of product names. O(m*n + nlogn) - O(1)
Optimized approach: I O(w) - S O(m)
'''
class Trie(object):
    def __init__(self):
        self.root = TrieNode()


    def insert(self, data):
        node = self.root
        idx = 0
        for char in data:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            if len(node.search_words) < 3:
                node.search_words.append(data)
            idx += 1

    def search(self, word):
        result, node = [], self.root
        for i, char in enumerate(word):
            if char not in node.children:
                temp = [[] for _ in range(len(word) - i)]
                return result + temp
            else:
                node = node.children[char]
                result.append(node.search_words[:])
        return result


def suggested_products(products, search_word):
    products.sort()
    trie = Trie()
    for x in products:
        trie.insert(x)
    return trie.search(search_word)

## Replace Words 
'''
You’re given a sentence consisting of words and a dictionary of root words. Your task is to find all the words in the sentence whose prefix matches with a root word present in the dictionary, and then to replace each matching word with the root word.
If a word in a sentence matches more than one root word in the dictionary, replace it with the shortest matching root word, and if the word doesn’t match any root word in the dictionary, leave the word unchanged. Return the modified sentence as the output.
Flow:
- Create a trie and iterate over each word from the dictionary to store them in the trie.
- For each word in the sentence, check whether any initial sequence of characters matches a word in the trie.
- If one or more matching words are found, replace it with the shortest matching word from the dictionary.
- Else, move to the next words from the sentence.
- Return the changed sentence after processing all the words in it.
Naive approach: For each word in the sentence, we’ll compare it with each dictionary word in turn. If a match is found, we compare its length with the length of the shortest match found up to that point. If it’s shorter, we update the index that keeps track of the shortest match. After checking the entire dictionary, we use the smallest matching word to replace the word in the sentence. O(dw), where d is the number of words in the dictionary and wi is the number of letters of the word in the sentence. O(dws) where s is the number of words in the sentence.
Optimized approach: O((d + s)l) - O(m)
'''
def replace_words(sentence, dictionary):
    trie = Trie()
    # iterate over the dictionary words, and
    # insert them as prefixes into the trie
    for prefix in dictionary:
        trie.insert(prefix)
    # split and assign each word from the sentence to new_list
    # this new_list is intended to return the final sentence
    # after all possible replacements have been made
    new_list = sentence.split()

    # iterate over all the words in the sentence
    for i in range(len(new_list)):
        # replace each word in the new_list with the
        # smallest word from dictionary
        new_list[i] = trie.replace(new_list[i])

    # after replacing each word with the matching dictionary word,
    # join them with a space in between them to reconstruct the sentence
    return " ".join(new_list)

## Design Add and Search Words Data Structure
'''
Design a data structure that supports the functions that add new words, find a string matches a previously added string, and return all words in the data structure.
Let’s call this data structure the WordDictionary class. Here’s how it should be implemented:
- Init(): This function will initialize the object.
- Add word(word): This function will add a word to the data structure so that it can be matched later.
- Search word(word): This function will return TRUE if there is any string in the WordDictionary object that matches the query word. Otherwise, it will return FALSE. If the query word contains dots, ., then it should be matched with every letter. For example, the dot in the string “.ad” can have 26 possible search results like “aad”, “bad”, “cad”, and so on.
- Get words(): This function will return all of the words present in the WordDictionary class.
Flow:
- Begin from the root node and iterate over the string one character at a time.
- Calculate the index of the character between 1 and 26.
- Place the character at its correct index in the dictionary.
- Set the boolean variable to TRUE when the word is complete.
Naive approach:  A naive approach would be to use a hash map. However, this solution would be inefficient when searching for all strings with a common prefix. Furthermore, once the hash map increases in size, there will be increased hash collisions. O(m*n)
Optimized approach: 
'''
class TrieNode():
  def __init__(self):
    self.nodes = []
    self.complete = False
    for i in range (0, 26):
      self.nodes.append(None)

class WordDictionary:
    # initialise the root with trie_node() and set the can_find boolean to False
    def __init__(self):
        self.root = TrieNode()
        self.can_find = False

    # get all words in the dictionary
    def get_words(self):
        wordsList = []
        # return empty list if there root is NULL
        if not self.root:
            return []
        # else perform depth first search on the trie
        return self.dfs(self.root, "", wordsList)

    def dfs(self, node, word, wordsList):
        # if the node is NULL, return the wordsList
        if not node:
            return wordsList
        # if the word is complete, add it to the wordsList
        if node.complete:
            wordsList.append(word)

        for j in range(ord('a'), ord('z')+1):
            n = word + chr(j)
            wordsList = self.dfs(node.nodes[j - ord('a')], n, wordsList)
        return wordsList

    # adding a new word to the dictionary
    def add_word(self, word):
        n = len(word)
        cur_node = self.root
        for i, val in enumerate(word):
            # place the character at its correct index in the nodes list
            index = ord(val) - ord('a')
            if cur_node.nodes[index] is None:
                cur_node.nodes[index] = TrieNode()
            cur_node = cur_node.nodes[index]
            if i == n - 1:
                # if the word is complete, it's already present in the dictionary
                if cur_node.complete:
                    print("\tWord already present")
                    return
                # once all the characters are added to the trie, set the complete variable to True
                cur_node.complete = True
        print("\tWord added successfully!")

    # searching for a word in the dictionary
    def search_word(self, word):
        # set the can_find variable as false
        self.can_find = False
        # perform depth first search to iterate over the nodes
        self.depth_first_search(self.root, word, 0)
        return self.can_find

    def depth_first_search(self, root, word, i):
        # if word found, return true
        if self.can_find:
            return
        # if node is NULL, return
        if not root:
            return
        # if there's only one character in the word, check if it matches the query
        if len(word) == i:
            if root.complete:
                self.can_find = True
            return

        # if the word contains ".", match it with all alphabets
        if word[i] == '.':
            for j in range(ord('a'), ord('z') + 1):
                self.depth_first_search(root.nodes[j - ord('a')], word, i + 1)
        else:
            index = ord(word[i]) - ord('a')
            self.depth_first_search(root.nodes[index], word, i + 1)

## Word Search II
'''
You are given a list of strings that you need to find in the 2-D grid. Neighboring characters can be combined to form strings. Remember that the diagonals aren’t included in neighboring characters— only up, down, right, and left neighbors can be considered. The solution should return a list containing the strings from the input list that were found in the grid.
Flow:
- Insert all the input strings in the trie.
- Search through the grid cells, looking in all four possible adjacent directions to see if there is a string in the input strings that starts with a letter in the cell.
- Using trie, for each cell in the grid, check if the traversed sequence of letters matches any string in the input strings.
- If the sequence matches a string in the input strings, include it in the result list.
- Return the result list.
O(n*3^l) - O(m + n)
'''
class TrieNode():
    def __init__(self):
        self.children = {}
        self.is_string = False

class Trie():
    def __init__(self):
        self.root = TrieNode()
    
    # Function to insert a string in the trie
    def insert(self, string_to_insert):
        node = self.root
        for c in string_to_insert:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children.get(c)
        node.is_string = True
    
    # Function to search a string from the trie
    def search(self, string_to_search):
        node = self.root
        for c in string_to_search:
            if c not in node.children:
                return False
            node = node.children.get(c)
        return node.is_string
    
    # Function to search prefix of strings
    def starts_with(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children.get(c)
        return True

    # Function to delete the characters in the searched word that are not shared
    def remove_characters(self, string_to_delete):
        node = self.root
        child_list = []
    
        for c in string_to_delete:
            child_list.append([node, c])
            node = node.children[c]
        
        for pair in reversed(child_list):
            parent = pair[0]
            child_char = pair[1]
            target = parent.children[child_char]

            if target.children:
                return
            del parent.children[child_char]

def find_strings(grid, words):
    trie_for_words = Trie()
    result = []
    # Inserting strings in the dictionary
    for word in words:
        trie_for_words.insert(word)
    # Calling dfs for all the cells in the grid
    for j in range(len(grid)):
        for i in range(len(grid[0])):
            dfs(trie_for_words, trie_for_words.root, grid, j, i, result)       
    return result

def dfs(words_trie, node, grid, row, col, result, word=''):
    # Checking if we found the string
    if node.is_string:
        result.append(word)
        node.is_string = False
        # remove the characters in the word that are not shared
        words_trie.remove_characters(word)
    
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        char = grid[row][col]
        # Getting child node of current node from Trie
        child = node.children.get(char)
        # if child node exists in Trie
        if child is not None:
            word += char
            # Marking it as visited before exploration
            grid[row][col] = None
            # Recursively calling DFS to search in all four directions
            for row_offset, col_offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(words_trie, child, grid, row + row_offset, col + col_offset, result, word)

            # Restoring state after exploration
            grid[row][col] = char
    
### *** Hash Maps
'''
Hash maps store data in the form of key-value pairs. They are similar to arrays because array values are stored against numeric keys. These keys are known as indexes. We don’t get to pick the value of these keys as they are always sequential integers starting from 0. Therefore, if we want to find an element within an array and don’t know it’s index, we’ll have to search the entire array which, in the worst case, will take O(N) time.
On the contrary, hash maps allow us to have flexible keys. Each key is unique and is mapped against a value. Therefore, we can look up its value in O(1) time.
'''

## Design HashMap
'''
Design a hash map without using the built-in libraries. We only need to cater integer keys and integer values in the hash map. Return NULL if the key doesn’t exist.
It should support the following three primary functions of a hash map:
- Put(key, value): This function inserts a key and value pair into the hash map. If the key is already present in the map, then the value is updated. Otherwise, it is added to the bucket.
- Get(key): This function returns the value to which the key is mapped. It returns −1, if no mapping for the key exists.
- Remove(key): This function removes the key and its mapped value.
Flow:
- Select a prime number (preferably a large one) as the key space.
- Initialize an array with empty buckets (empty arrays). The number of buckets in the array should be equal to the specified value of the key space variable.
- Generate a hash key by taking the modulus of the input key with the key space variable.
- Perform the appropriate function (Put(), Get(), Remove()).
'''
# A class implementation of the bucket data structure
class Bucket:
    # Initialize buckets here
    def __init__(self):
        self.buckets = []

    # get value from bucket
    def get(self, key):
        for (k, v) in self.bucket:
            if k == key:
                return v
        return -1

    # put value in bucket
    def update(self, key, value):
        found = False
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key, value)
                found = True
                break

        if not found:
            self.bucket.append((key, value))

    # delete value from bucket
    def remove(self, key):
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]

class MyHashMap():
    # Initialize hash map here
    def __init__(self, key_space):
        # It’s better to have a prime number, so there's less collision
        self.key_space = key_space
        self.buckets = [Bucket()] * self.key_space

    # Function to add value of a given key
    # hash map at the relevant hash address
    def put(self, key, value):
        if key== None or value == None:
            return
            
        hash_key = key % self.key_space
        self.buckets[hash_key].update(key, value)

    # Function to fetch corresponding value of a given key
    def get(self, key):
        if key == None:
            return -1
        hash_key = key % self.key_space
        return self.buckets[hash_key].get(key)

    # Function to remove corresponding value of a given key
    def remove(self, key):
        hash_key = key % self.key_space
        self.buckets[hash_key].remove(key)

## Fraction to Recurring Decimal
'''
Given the two integer values of a fraction, numerator and denominator, implement a function that returns the fraction in string format. If the fractional part repeats, enclose the repeating part in parentheses.
Flow:
- Declare a result variable to store the result in the form of a string. In addition, declare a hash map to store the remainder as the key and the length of the result variable as its corresponding value.
- Calculate the quotient and remainder from the given numerator and denominator.
- Check if the remainder is 0, then return the result.
- If the remainder is not 0, then append the dot “.” to the result.
- Start a loop until the remainder is 0 and at every time check the remainder in the hash map. If the remainder already exists in the hash map, then create the recurring decimal from the fraction. If the remainder does not exist in the hash map, put it in the hash map.
Naive approach: use the array to store the remainders. Every time we calculate the remainder we can check if it already exists in the array by searching in the array. This approach will use a nested loop. The outer loop will calculate the remainder and the inner loop will search in the array for the remainder. O(|d|^2)
Optimized approach: O(|d|)
'''
def fraction_to_decimal(numerator, denominator):
    result, hash_map = "", {}
    # if the numerator is 0, then return 0 in the string
    if numerator == 0:
        return '0'

    # If the numerator or denominator is negative, then append "-" to the result
    if (numerator < 0) ^ (denominator < 0):
        result += '-'

        # Make the numerator and denominator positive after adding "-" to the result
        numerator = abs(numerator)
        denominator = abs(denominator)

    # Calculate the quotient
    quotient = numerator / denominator
    # Calculate the remainder
    remainder = (numerator % denominator) * 10
    # Append the quotient part in the result
    result += str(int(quotient))
    # if the remainder is 0, then return the result
    if remainder == 0:
        return result
    else:
    # Append . before the right part
        result += "."
    # Right side of the decimal point
        while remainder != 0:
        # if digits are repeating, it means the remainder is already present in the hashmap
            if remainder in hash_map.keys():
                # Convert fraction to recurring decimal
                beginning = hash_map.get(remainder)
                left = result[0: beginning]
                right = result[beginning: len(result)]
                result = left + "(" + right + ")"
                return result

        # Otherwise put the remainder in the hashmap
            hash_map[remainder] = len(result)
            quotient = remainder / denominator
            result += str(int(quotient))
            remainder = (remainder % denominator) * 10
        return result

## Logger Rate Limiter
'''
For the given stream of message requests and their timestamps as input, you must implement a logger rate limiter system that decides whether the current message request is displayed. The decision depends on whether the same message has already been displayed in the last S seconds. If yes, then the decision is FALSE, as this message is considered a duplicate. Otherwise, the decision is TRUE.
Flow:
- Use all incoming messages as keys and their respective timestamps as values, to form key-value pairs to store them in a hash map.
- When a request arrives, use the hash map to check if it’s a new request or a repeated request. If it’s a new request, accept it and add it to the hash map.
- If it’s a repeated request, check if S seconds have passed since the last request with the same message. If this is the case, accept it and update the timestamp for that specific message in the hash map.
- If the repeated request has arrived before the time limit has expired, reject it.
Naive approach: The problem we’re trying to solve here implies that there’s a queue of messages with timestamps attached to them. Using this information, we can use a queue data structure to process the incoming message requests. In addition to using a queue, we can use a set data structure to efficiently identify and remove duplicates. Our naive approach is an event-driven algorithm, where every incoming message prompts us to identify and remove all those messages from the queue whose timestamps are more than S seconds older than the timestamp of the new message. Whenever we remove a message from the queue, we also remove it from the set. After performing this time limit expiry check, we can be certain that none of the messages in the queue and in the set are more than S seconds older than the new message. Now, if this new message is present in the set, it’s a duplicate and we return FALSE. Otherwise, we add it to the message queue and to the message set, and return TRUE. O(n)
Optimized approach: O(1) - O(n)
'''
class RequestLogger:

    # initailization of requests hash map
    def __init__(self, time_limit):
        self.requests = {}
        self.limit = time_limit

    # function to accept and deny message requests
    def message_request_decision(self, timestamp, request):

        # checking whether the specific request exists in
        # the hash map or not if it exists, check whether its
        # time duration lies within the defined timestamp
        if request not in self.requests or timestamp - self.requests[request] >= self.limit:

            # store this new request in the hash map, and return true
            self.requests[request] = timestamp
            return True

        else:
            # the request already exists within the timestamp
            # and is identical, request should
            # be rejected, return false
            return False

## Next Greater Element
'''
Let's first define the "next greater element" of some element x in an array of integers as the first element we encounter to the right of x (that is, whose index is greater than the index of x) whose value is also greater than x. In mathematical terms, y is the next greater element of x, if and only if: 
- y > x
- index of y > index of x 
- the first two conditions don't hold true for any other element z , where index of z < index of y .
You are given two distinct integer arrays, nums1 and nums2, where nums1 is a subset of nums2.
For each index i , where 0 ≤ i < nums1.length , find the index j such that nums1[i] = nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, the answer to this query is −1.
Compose and return an array ans of the same length as that of nums1, such that each value ans[i] is the next greater element of nums1[i] in nums2.
Flow:
- Create a hash map that stores the indexes and the values of all the elements in the nums1 array. Also create the ans array. It should have the same size as nums1 and set each slot to -1.
- Traverse the nums2 array and check if the current element is present in the hash map.
- If the specific number is not found, we skip to the next element of nums2. Otherwise, note the position of the matching element in nums2 and find the next greater element.
- If the next greater element exists, store it in the ans array at the correct position. Otherwise retain -1 at that index.
Naive approach: The naive approach is to select an element of the nums1 array and then search for when it occurs in the nums2 array. If the element is found, we look for the occurrence of the next greater element. If we obtain the next greater element, we store it in the ans array in the slot corresponding to the element in nums1. If we cannot find the next greater element, we simply store −1 in the ans array. We repeat the algorithm above until we have traversed all the elements of the nums1 array. O(n1*n2) - O(1)
Optimized solution: O(n21 + n22) = O(n22) - O(n1)
'''
def next_greater_element(nums1, nums2):

    nums1_indexed = {}  # Initializing the hash map
    for i, n in enumerate(nums1):
        nums1_indexed[n] = i
    ans = [-1] * len(nums1)

    
    for i in range(len(nums2)):
        index = nums1_indexed.get(nums2[i])
        if isinstance(index, type(None)):   # cannot use in operator as index may contain 0
            # If the value doesn't show up in the nums1 array, we
            # move on to the next element in nums2 array.
            continue
        for j in range(i+1, len(nums2)):
            if nums2[j] > nums2[i]:
                ans[index] = nums2[j]
                # Got the first missing value, we don't need
                # to continue any further
                break
    return ans

## Isomorphic Strings
'''
Given two strings, check whether two strings are isomorphic to each other or not.  Two strings are isomorphic if a fixed mapping exists from the characters of one string to the characters of the other string. For example, if there are two instances of the character "a"  in the first string, both these instances should be converted to another character (which could also remain the same character if "a" is mapped to itself) in the second string. This converted character should remain the same in both positions of the second string since there is a fixed mapping from the character "a" in the first string to the converted character in the second string.
Flow:
- Create two hash maps. One to store mapping from string1 to string2 and another to store mapping from string2 to string1.
- Before storing the mapping of characters in the hash maps, first check if the character is present in its hash map.
- If the character is already in the hash map, and is mapped to a different character in the hash map than the character to be mapped, the algorithm terminates and FALSE is returned.
- If all the mappings are valid in both the hash maps, TRUE is returned.
O(n) - O(1)
'''
def is_isomorphic(string1, string2):

    # Initializing the hashmaps
    map_str1_str2 = {}
    map_str2_str1 = {}

    for i in range(len(string1)):
        char1 = string1[i]
        char2 = string2[i]

        # returning false if char_1 in string1 exist in hashmap
        # and the char_1 has different mapping in hashmap
        if char1 in map_str1_str2 and map_str1_str2[char1] != char2:
            return False

        # returning false if char_2 in string2 exist in hashmap
        # and the char_2 has different mapping in hashmap
        if char2 in map_str2_str1 and map_str2_str1[char2] != char1:
            return False

        # mapping of char of one string to another and vice versa
        map_str1_str2[char1] = char2
        map_str2_str1[char2] = char1

    return True

### *** Knowing What to Track
'''
The knowing what to track pattern is mostly used to find the frequency count of the data. It’s able to avoid the O(n^2) time by simplifying the problem and identifying certain key numeric properties in the given data.
This pattern is often used with an array and strings, and helps solve problems that require counting data frequencies or returning boolean values (TRUE/FALSE). More often than not, the problems that can be solved using this pattern include comparison and checks
'''

## Palindrome Permutation
'''
For a given string, find whether or not a permutation of this string is a palindrome. You should return TRUE if such a permutation is possible and FALSE if it isn’t possible.
Flow:
- Traverse the input string starting from the first character.
- Populate a hash map with the characters in the string, along with the frequency of occurrence of each character.
- Traverse the hash map to get the count of characters with an odd number of occurrences.
- If the count exceeds 1, no palindromic permutation exists.
Naive approach: ompute all possible permutations of a given string and then iterate over each permutation to see if it’s a palindrome. We can use two pointers to traverse the computed permutation from the beginning and the end simultaneously, comparing each character at every step. If the two pointers point to the same character until they cross each other, the string is a palindrome. O(n!) + O(n) | O(n!)
Optimized approach: O(n^2) ~ O(n) - O(n1)
'''
def permute_palindrome(st):
    # Create a hashmap to keep track of the characters and their occurrences
    frequencies = {}
    index = 0
    for i in st:
        index += 1
        if i in frequencies:
            frequencies[i] += 1
        else:
            frequencies[i] = 1
    # Traverse the hashmap and update the count by 1, whenever a
    # character with odd number of occurences is found.
    count = 0
    for ch in frequencies.keys():
        if frequencies[ch] % 2:
            count += 1
    # If the count is smaller than or equal to 1 then the permutation exists,
    # otherwise does not
    if count <= 1:
        return True
    else:
        return False

## Design Tic-Tac-Toe
'''
Suppose that two players are playing a tic-tac-toe game on an n×n board. They’re following specific rules to play and win the game.
- A move is guaranteed to be valid if a mark is placed on an empty block.
- No more moves are allowed once a winning condition is reached.
- A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
Your task is to implement a TicTacToe class, which will be used by two players to play the game and win fairly.
Keep in mind the following functionalities that need to be implemented:
- The TicTacToe class, which declares an object to create the board.
- Init (n), which initializes the object of TicTacToe to create the board of size n.
- Move (row, col, player) indicates that the player with ID player plays at the board’s cell (row,col). The move is guaranteed to be a valid move. At each move, this function returns the player ID if any player wins and returns 0 if no one wins.
Flow:
- Initialize arrays, rows, and cols of size n with 0 at each index for each player.
- If rows[i] or cols[j] is equal to n, the player wins.
- Initialize integer variables, diagonal and anti-diagonal, for each player.
- Increment the diagonal and anti-diagonal integers whenever a player marks a cell on each of the diagonal.
- If the diagonal or anti-diagonal is equal to n, the player wins.
Naive approach: check on every move if the current player won the game. Each player marks the cell at row and col to make a move on the board. One player can win the game using the following four ways:
- Player marked the complete row.
- Player marked the complete column.
- Player marked all the diagonal cells, starting from the top left corner to the bottom right corner of the board.
- Player marked all the anti-diagonal cells, starting from the top right corner to the bottom left corner of the board.
We mark the row and col on the board for the current with the player’s ID on every move. Then, we check the conditions to check if the current player wins.
- Check the condition for the row by iterating over all the columns but keeping the row index constant.
- Check the condition for the col by iterating over all the rows but keeping the column index constant.
- Check the condition for diagonal by iterating the board and keeping the same row and column index values.
- Check the condition for anti-diagonal by iterating the board and keeping the column equal to n−row−1.
O(n) - O(n^2)
Optimized approach: O(n) - O(n)
'''
class TicTacToe:
    # TicTacToe class contains rows, cols, diagonal,
    # and anti_diagonal to create a board.
    # Constructor is used to create n * n tic - tac - toe board.
    def __init__(self, n):
        self.rows = [0] * (n)
        self.cols = [0] * (n)
        self.diagonal = 0
        self.anti_diagonal = 0
        self.board = []
        for i in range(n):
            st = ""
            for j in range(n):
                st += "-"
            self.board.append(st)

    # Move function will allow the players to play the game
    # for given row and col.
    def move(self, row, col, player):
        current_player = -1
        if player == 1:
            current_player = 1

        self.rows[row] += current_player
        self.cols[col] += current_player

        if row == col:
            self.diagonal += current_player

        if col == (len(self.cols) - row - 1):
            self.anti_diagonal += current_player

        st = list(self.board[row])
        st[col] = str(current_player)
        
        st = list(self.board[row])
        if current_player == 1:
            st[col] = "O"
        else:
            st[col] = "X"
        self.board[row] = "".join(st)

        n = len(self.rows)

        if abs(self.rows[row]) == n or abs(self.cols[col]) == n or abs(self.diagonal) == n or abs(self.anti_diagonal) == n:
            return player
        return 0

## Group Anagrams
'''
Given a list of words or phrases, group the words that are anagrams of each other. An anagram is a word or phrase formed from another word by rearranging its letters.
Flow:
- Initialize a hash map to store key-value pairs for the strings’ frequency and its anagrams, respectively. The key will be a character list of length 26, initialized to all 0s, and the value will be an array of anagrams.
- Start traversing the list of strings. Within each string in the list, traverse each character of the string. Reset the character list to all 0s before beginning this second traversal.
- Calculate the index of each letter of the string through the ACSII value of the character. In the character list, increment the value at this index by 1.
- After a string has been traversed, check whether the current character list is present as a key in the hash map.
- If the current character list is present as a key in the hash map, add the string to the list of anagrams corresponding to this key.
- Otherwise, add a new key-value pair to the hash map with the current character array as the key and the traversed string as the value in an array.
- Repeat this process until all the strings have been traversed.
Naive approach: ort all strings first and then compare them to check whether they are identical. The idea is that if two strings are anagrams, sorting them both will always make them equal to each other. We’ll initialize a hash map to store the anagrams where the key represents the sorted string and the value represents the array of anagrams corresponding to that key. We’ll run a loop on the given list of strings. On each iteration, we will sort the current string. We’ll then check if the sorted string is present as a key in the hashmap. If it is, we’ll append the original, unsorted string to the array corresponding to that key. Otherwise, we’ll add the new key-value pair to the hashmap. At the end of the traversal, the hashmap will contain all the groups of anagrams. O(nllog(l)) - O(1)
Optimized approach: O(n*k) - O(n*k)
'''
def dict_to_lst(subset):
        result = []
        if subset == []:
            return [[]]
        for dic in subset:
            result.append(list(dic))
        return result


def group_anagrams(strs):
    res = {}  # Hashmap for count
    for s in strs:
        count = [0] * 26  # A place for every single alphabet in our string
        # We will compute the frequency for every string
        for i in s:
            # Calculating the value from 1 to 26 for the alphabet
            index = ord(i) - ord('a')
            count[index] += 1  # Increasing its frequency in the hashmap
        # Each element in this tuple represents the frequency of an
        # English letter in the corresponding title
        key = tuple(count)
        if key in res:
            res[key].append(s)
        else:
            # We add the string as an anagram if it matched the content
            # of our res hashmap
            res[key] = [s]
    return dict_to_lst(res.values())

## Maximum Frequency Stack
'''
Design a stack-like data structure. You should be able to push elements to this data structure and pop elements with maximum frequency.
You’ll need to implement the FreqStack class that should consist of the following:
- Init(): This is a constructor used to declare a frequency stack.
- Push(value): This is used to push an integer data onto the top of the stack.
- Pop(): This is used to remove and return the most frequent element in the stack.
Flow:
- Create a hash map or a dictionary to store frequencies of all elements.
- Iterate over the input array and store its frequency in a hash map or dictionary. The corresponding frequency value is a stack containing all the elements with that frequency.
- After all elements are pushed onto the stack, start removing the most frequent elements.
- While removing elements, also decrease their frequency count from the hash map or dictionary.
- When there’s a case where two or more elements have the same frequency, pop the latest element that was pushed onto the stack.
Naive approach: using a heap data structure, specifically a max heap. This approach involves maintaining a hash map to keep track of the frequency of each element in the stack and a counter variable to keep track of the last inserted element.
To push elements onto the max-heap, we need to follow these steps:
- Check if the element is already present in a heap. If it is, increment its frequency in the hash map. Otherwise, set its frequency to 1.
- Increment the counter by 1.
- Add an array containing the element, its frequency, and the counter to the max-heap.
To pop elements from the max-heap, we need to follow these steps:
- Remove the most frequently occurring element from the max-heap.
- Decrement the frequency of that element in the hash map.
O(nlogn) - O(n)
Optimized approach:This code solves the problem by maintaining a stack that tracks the frequency of elements and retrieves the most frequently occurring element efficiently. It uses two dictionaries to keep track of the frequency of each element and the elements associated with a given frequency. When a new element is pushed onto the stack, its frequency is incremented, and it is added to the list of elements corresponding to that frequency. The maximum frequency is updated as needed. When a pop operation is requested, the stack retrieves the element with the highest frequency and decrements its frequency. If there are multiple elements with the same highest frequency, the most recently added element is removed first.
O(1) - O(n)
'''
from collections import defaultdict

# Declare a FreqStack class containing frequency and group hashmaps
# and maxFrequency integer
class FreqStack:

    # Use constructor to initialize the FreqStack object
    def __init__(self):
        self.frequency = defaultdict(int)
        self.group = defaultdict(list)
        self.max_frequency = 0

    # Use push function to push the value into the FreqStack
    def push(self, value):
        # Get the frequency for the given value and
        # increment the frequency for the given value
        freq = self.frequency[value] + 1
        self.frequency[value] = freq

        # Check if the maximum frequency is lower that the new frequency
        # of the given show
        if freq > self.max_frequency:
            self.max_frequency = freq

        # Save the given showName for the new calculated frequency
        self.group[freq].append(value)

    def pop(self):
        value = ""

        if self.max_frequency > 0:
            # Fetch the top of the group[maxFrequency] stack and
            # pop the top of the group[maxFrequency] stack
            value = self.group[self.max_frequency].pop()

            # Decrement the frequency after the show has been popped
            self.frequency[value] -= 1

            if not self.group[self.max_frequency]:
                self.max_frequency -= 1
        else:
            return -1

        return value

## First Unique Character in a String
'''
For a given string of characters, s, your task is to find the first non-repeating character in it and return its index. Return −1 if there’s no unique character in the given string.
Flow: 
- Iterate over the length of input string.
- For each character in the string, check if it exists in the hashmap. If it does, increment its value by one. Otherwise, add the character in the hashmap and its initial count of 1 as a new key-value pair.
- Traverse the hashmap to find the first occurrence of a non-repeating value.
- If any non-repeating character exists, return its index. Else, return -1.
O(n) - O(1)
'''
def first_unique_char(s):

    character_count = {}  # declaring a hash map
    string_length = len(s)  # storing the length of the input string

    # loop to iterate over the length of input string
    for i in range(string_length):
        # check if the character exists in the hash map
        if s[i] in character_count:
            # if the character already exists, 
            # increase the counter by adding +1
            character_count[s[i]] += 1
        else:
            # if the character doesn't exists, 
            # set the count of letter to 1
            character_count[s[i]] = 1

    # this loop will check the first occurrence of a letter whose count is of 1
    for i in range(string_length):
        # the first character to have a count of 1 should be returned
        if character_count[s[i]] == 1:
            return i

    # return -1 if all occurrences of letters have a count greater than 1
    return -1

## Find All Anagrams in a String
'''
Given two strings, a and b, return an array of all the start indexes of b's anagrams in a. We may return the answer in any order.
An anagram is a word or phrase created by rearranging the letters of another word or phrase while utilizing each of the original letters exactly once.
Flow:
- If length of string b is more than length of string a, return an empty list.
- Initialize two hash maps to store the frequency of different characters occurring in our strings.
- Start populating the hash maps based on characters found in the strings.
- Compare the two hash maps while traversing string a, and if an anagram is found, return the index of the element we started our search process from.
- Adjust the size of the window we searched and check if any more indexes exist where we can find string b from.
O(n + m) - O(n + m)
'''
def find_anagrams(a,b):
  if len(b) > len(a): #If len of b is more than len of a, then there could be no anagram of b in a
    return []
  
  count_a = defaultdict(int) # Hash map for count of a
  count_b = defaultdict(int) # Hash map for count of b

  for i in range(len(b)): #Storing frequency of characters for length of b
    count_a[a[i]] += 1
    count_b[b[i]] += 1

  if count_a == count_b: #A specific case where we add the first index if matched
    ans = [0]
  else:
    ans = []

  sliding_left = 0 # We declare the sliding window pointer
  for right in range(len(b), len(a)):
    count_a[a[right]] += 1
    count_a[a[sliding_left]] -= 1
  
    if count_a[a[sliding_left]] == 0:
      count_a.pop(a[sliding_left])
  
    sliding_left = sliding_left + 1
    if count_a == count_b:
      ans.append(sliding_left) #If all the characters from b with their respective frequency are found at this index, we append.

  return ans

### *** Union Find
'''
The union find pattern is used to group elements into sets based on a specified property. Each set is non-overlapping, that is, it contains unique elements that are not present in any other set. The pattern uses a disjoint set data structure such as an array, to keep track of which set each element belongs to.
Each set forms a tree data structure and has a representative element that resides at the root of the tree. Every element in this tree maintains a pointer to its parent. The representative’s parent pointer points to itself. If we pick any element in a set and follow its parent pointers, we’ll always reach the set representative.
'''


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

## Top K Elements - Kth Smallest Element in a BST
'''
Given the root node of a binary search tree and an integer value k, return the kth smallest value from all the nodes of the tree.
Flow:
- Traverse the BST in the inorder fashion.
- At each node, save its value in an array, building up a list of values in a tree, sorted in ascending order.
- Once the entire tree has been traversed, fetch the k−1th element from the list.
'''

## Modified Binary Search - Search in Rotated Sorted Array II
'''
You are required to find an integer t in an array arr of non-distinct integers. Return TRUE if t exists in the rotated, sorted array arr, and FALSE otherwise, while minimizing the number of operations in the search.
Flow:
- Declare two low and high pointers that will initially point to the first and last indexes of the array, respectively.
- Declare a middle pointer that will initially point to the middle index of the array. This divides the array into two halves.
- Check if the target is present at the position of the middle pointer. If it is, return True.
- If the first half of the array is sorted and the target lies in this range, update the high pointer to mid in order to search in the first half.
- Else, if the second half of the array is sorted and the target lies in this range, update the low pointer to mid in order to search in the second half.
- If the high pointer becomes greater or equal to the low pointer and we still haven’t found the target, return False.
'''

## Subsets - Find K-Sum Subsets
'''
Given a set of n positive integers, find all the possible subsets of integers that sum up to a number k.
Flow:
- Find all possible subsets of the set.
- Find the sum of the elements of each subset.
- If the sum for any subset equals k, then add this subset into the result list.
- Return the result list.
'''

## Greedy - Jump Game II
'''
In a single-player jump game, the player starts at one end of a series of squares, with the goal of reaching the last square. At each turn, the player can take up to s steps towards the last square, where s is the value of the current square. For example, if the value of the current square is 3 , the player can take either 3 steps, or 2 steps, or 1 step in the direction of the last square. The player cannot move in the opposite direction, that is, away from the last square. You’ve been provided with the nums integer array, representing the series of squares. You’re initially positioned at the first index of the array. Find the minimum number of jumps needed to reach the last index of the array. You may assume that you can always reach the last index.
Flow:
- Initialize three variables: farthest_jump, denoting the farthest index we can reach, current_jump, denoting the end index of our current jump, and jumps, to store the number of jumps. All three of these variables are set to 0.
- Traverse the entire nums array. On each ith iteration, update farthest_jump to the max of the current value of farthest_jump, and i + nums[i].
- If i is equal to current_jump, we have completed the current jump and can now prepare to take the next jump (if required). So we increment the jumps variable by 1 and set current_jump equal to farthest_jump.
- Otherwise, do not update either the jumps variable or the current_jump variable, since we haven’t yet completed the current jump.
- At the end of the traversal, the jumps variable will contain the minimum number of jumps required to reach the last index.
'''
## Backtracking - Sudoku Solver
'''
Given a 9 x 9 sudoku board, solve the puzzle by completing the empty cells. The sudoku board is only considered valid if the rules below are satisfied:
Each row must contain digits between 1–9, and there should be no repetition of digits within a row.
Each column must contain digits between 1–9, and there should be no repetition of digits within a column.
The board consists of 9 non-overlapping sub-boxes, each containing 3 rows and 3 columns. Each of these 3 x 3 sub-boxes must contain digits between 1–9, and there should be no repetition of digits within a sub-box.
Flow:
- Start iterating the board from the top left cell until we reach the first free cell.
- One by one, place all numbers between 1 and 9 in the current cell, if that number isn’t already present in the current row, column and 3x3 sub-box.
- Write down that number that is now present in the current row, column, and box.
- If we reach the last cell, that means we’ve successfully solved the sudoku.
- Else, we move to the next cell.
- Backtrack if the solution is not yet present, and remove the last number from the cell.
'''

## Backtracking - Matchsticks to Square
'''
Given an integer array, matchsticks, where matchsticks[i] is the length of the ith matchstick. Use every single matchstick to create a square. No stick should be broken, although they can be connected, and each matchstick can only be used once.
Return TRUE if we can make this square and FALSE otherwise.
Flow:
- If the number of matchsticks is less than 4, return FALSE.
- If the sum of all values in matchsticks is less than 4, return FALSE.
- If the sum of all values in matchsticks is not a multiple of 4, return FALSE.
- Sort matchsticks in descending order and set the length of one side to 1/4th of the sum of the values of matchsticks.
- Choose a matchstick for one of the sides and check if it’s equal to the required length of one side.
- If it’s equal to the required length, try constructing another side using the remaining matchsticks.
- If it’s less than the required length, try to complete the side.
- If it’s greater than the required length, return FALSE.
'''

## Dynamic Programming - Climbing Stairs
'''
You are climbing a staircase. It takes n steps to reach the top. Each time, you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
Flow:
- Find the number of ways to climb 2 stairs.
- Find the number of ways to climb 3 stairs.
- Find the number of ways to climb 5 stairs.
- Find the number of ways to climb (n-2) stairs.
- Find the number of ways to climb n stairs.
'''

##

## DFS - Maximum Depth of Binary Tree
'''
Given the root of a binary tree, return its maximum depth. A binary tree’s maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
Flow:
- Initialize a counter to track the maximum depth seen so far and a counter for the depth of the current branch.
- If the current node is NULL, return the counter.
- Else, if the current node exists, check the depth of the left subtree and the depth of the right subtree.
- Compare the depths of the left and right subtrees, and select the greater of the two. Add 1 and update this as the depth of this branch.
- If the depth of the current branch exceeds the maximum depth seen so far, update the maximum depth.
- When all branches have been explored, return the maximum depth seen so far.
'''

## BFS - Connect All Siblings of a Binary Tree
'''
The task is to connect all nodes in a binary tree. Connect them from left to right so that the next pointer of each node points to the node on its immediate right. The next pointer of the right-most node at each level should point to the first node of the next level in the tree.
Each node in the given binary tree for this problem includes a next pointer, along with the left and right pointers. Your solution must set the next pointer to connect the same level nodes to each other and across levels.
Flow:
- Initialize two pointers, current and last, to root.
- If the current node has a left child, set the next pointer of the last node to this left child and set the last pointer to this left child.
- Else, if the current node has a right child,set the next pointer of the last node to this right child, and set the last pointer to this right child.
- Update the current node to the current node’s next node.
- Return the next node of the desired node.
'''

## Trie - Lexicographical Numbers
'''
Given an integer value n, write a function that returns all the numbers in the range 1 to n in lexicographical order.
Flow:
- Insert numbers from 1 to n in the trie. Each number should be split into digits by the trie and saved as trie nodes.
- Traverse Trie structure in preorder traversal format and append each trie node to result array by prefixing its parent nodes till the root.
- Return the result array as it contains the lexicographical order of n numbers.
'''

## Longest Palindrome
'''
Given a string, pal_string, consisting of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.
Flow:
- Iterate through the input string to count the occurrences of each character and to store these in a hash map.
- Iterate through the counts in the hash map, and check whether each value is even or odd.
- If a count is even, add this value to pal_string–that is, the length of the longest possible palindrome.
- Otherwise, add/subtract 1 from this odd value and add that to pal_string. Also, set a flag to indicate that an odd-valued count was encountered.
- After going through all the character counts in the hash map, if the flag for odd-valued counts is set, add it to pal_string.
'''

## Knowing What to Track - Ransom Note
'''
Given two strings, “ransom note” and “magazine”, check if the ransom note can be constructed using the letters from the magazine string. Return TRUE if a ransom note can be constructed. Otherwise, return FALSE.
Flow:
- Fill the hash map using the magazine string.
- Keep the count of every unique character in the hash map.
- Iterate the ransom note string.
- Check if the character is present in the hash map. If the character is not present, return FALSE.
- Check if the count of the character is not 0 in the hash map. If 0, return FALSE.
- If we find the character in the hash map with no 0 count, we decrement the character count in the hash map.
'''