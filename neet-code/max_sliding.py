import collections

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        q = collections.deque()  # index
        l = r = 0
        # O(n) O(n)
        while r < len(nums):
            # pop smaller values from q
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            # remove left val from window
            if l > q[0]:
                q.popleft()

            if (r + 1) >= k:
                output.append(nums[q[0]])
                l += 1
            r += 1

        return output

"""
Brute force: O(k * (n - k))
Better: no repeated comparisons - shift only one --> using deque O(n)
"""

# Grokking 
def find_max_sliding_window(nums, window_size):
    result = []
    # Initializing doubly ended queue for storing indices
    window = deque()
    
    # If window_size is greater than the array size,
    # set the window_size to nums.size()
    if window_size > len(nums):
        window_size = len(nums)

    # Find out first maximum in the first window
    for i in range(window_size):
        # For every element, remove the previous smaller elements from window
        while window and nums[i] >= nums[window[-1]]:
            window.pop()
        # Add current element at the back of the queue
        window.append(i)
    # Appending the largest element in the window to the result
    result.append(nums[window[0]])
    for i in range(window_size, len(nums)):
        # remove all numbers that are smaller than current number
        # from the tail of list
        while window and nums[i] >= nums[window[-1]]:
            window.pop()
        # Remove first index from the window deque if
        # it doesn't fall in the current window anymore
        if window and window[0] <= (i - window_size):
            window.popleft()
        window.append(i)
        # Add current element at the back of the queue
        result.append(nums[window[0]])
    return result


def main():
    target_list = [3, 3, 3, 3, 2, 4, 3, 2, 3, 18]
    nums_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                 [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                 [1, 5, 8, 10, 10, 10, 12, 14, 15, 19, 19, 19, 17, 14, 13, 12, 12, 12, 14, 18, 22, 26, 26, 26, 28, 29, 30],
                 [10, 6, 9, -3, 23, -1, 34, 56, 67, -1, -4, -8, -2, 9, 10, 34, 67],
                 [4, 5, 6, 1, 2, 3],
                 [9, 5, 3, 1, 6, 3],
                 [2, 4, 6, 8, 10, 12, 14, 16],
                 [-1, -1, -2, -4, -6, -7],
                 [4, 4, 4, 4, 4, 4]]

    for i in range(len(nums_list)):
        print(i + 1, ".\tOriginal array:\t", nums_list[i], sep="")
        print("\tWindow size:\t", target_list[i])
        print("\n\tMax:\t\t",
              find_max_sliding_window(nums_list[i], target_list[i]))
        print("-"*100)


if __name__ == '__main__':
    main()