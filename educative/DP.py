### *** 0/1 Knapsack
## 0/1 Knapsack
'''
Suppose you have a list of weights and corresponding values for n items. You have a knapsack that can carry items up to a specific maximum weight, known as the capacity of the knapsack.
You want to maximize the sum of values of the items in your knapsack while ensuring that the sum of the weights of the items remains less than or equal to the knapsack’s capacity.
If all the combinations exceed the given knapsack’s capacity, then return 0.
Naive approach: find all combinations of items such that their combined weight is less than the capacity of the knapsack and the total value is maximized. In other words, we divide the problem into sub-problems, and for each item, if its weight is less than the capacity, we check whether we should place it in the knapsack.
O(2^n) - O(n)
Optimized solution:
Top-down solution:
In the recursive approach, the two variables that kept changing in each call were the total weight of the knapsack and the number of items we had. So, we’ll use these two variables to define each distinct subproblem. Therefore, we need a 2-D array to store these values and the result of any given subproblem when we encounter it for the first time.
O(n * cap) - O(n * cap)
---
Bottom-up solution: Here, we will create a 2-D array of size (n + 1) ∗ (capacity + 1). The row indicates the values of the available items, and the column shows the capacity of the knapsack at any given point.
We will initialize the array so that when the row or column is 0, the value in the table will also be 0. Next, we will check if the weight of an item is less than the capacity. If yes, we have two options: either we can add the item to the knapsack, or we can skip it. If we include the item, the optimal solution is the maximum of the two values. Otherwise, if the weight of an item is greater than the capacity, then we don’t include the item in the knapsack.
O(n * cap) - O(n * cap)
'''
# Naive
def find_knapsack(capacity, weights, values, n):
    #Base case
    if n == 0 or capacity == 0:
        return 0

    #check if the weight of the nth item is less than capacity then
    #either 
    # We include the item and reduce the weight of item from the total weight
    # or 
    # We don't include the item
    if (weights[n-1] <= capacity):
        return max(
            values[n-1] + find_knapsack(capacity-weights[n-1], weights, values, n-1),
            find_knapsack(capacity, weights, values, n-1)
            )
    #Item can't be added in our knapsack 
    #if it's weight is greater than the capacity
    else:
        return find_knapsack(capacity, weights, values, n-1)
    
# Optimized
# -- Top down
def find_knapsack(capacity, weights, values, n):
    dp = [[-1 for i in range(capacity + 1)] for j in range(n + 1)]
    return find_knapsack_value(capacity, weights, values, n, dp)

def find_knapsack_value(capacity, weights, values, n, dp):
    # Base case
    if n == 0 or capacity == 0:
        return 0
    
    #If we have solved it earlier, then return the result from memory
    if dp[n][capacity] != -1:
        return dp[n][capacity]
 
    #Otherwise, we solve it for the new combination and save the results in the memory
    if weights[n-1] <= capacity:
        dp[n][capacity] = max(
            values[n-1] + find_knapsack_value(capacity-weights[n-1], weights, values, n-1, dp),
            find_knapsack_value(capacity, weights, values, n-1, dp)
            )
        return dp[n][capacity]

# -- Bottom up
def find_knapsack(capacity, weights, values, n):
    #create a table to hold intermediate values
    dp = [[0 for i in range(capacity + 1)] for j in range(n + 1)]
 
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            #initialize the table with 0 when either the row or column is 0
            if i == 0 or j == 0:
                dp[i][j] = 0
            #check if the weight of an item is less than the capacity
            elif weights[i-1] <= j:
                dp[i][j] = max(values[i-1]+ dp[i-1][j-weights[i-1]], 
                              dp[i-1][j]
                            )
            #we don't include the item if the weight is greater than the capacity.                
            else:
                dp[i][j] = dp[i-1][j]
 
    return dp[-1][-1] #[n][capacity]

## Target Sum
'''
Given an array of positive integers arr and a target T, build an expression using these numbers by inserting a + or a − before each integer, and evaluating this expression. Find the total number of different expressions that evaluate to T.
Naive approach: find all expressions using the given numbers and then count the number of expressions that evaluate to the given target. In other words, we divide the problem into subproblems, and for each number, we place a + or − before it and generate new expressions.
O(n^2) - O(n)
---
Optimized solution
Top-down solution: we create a lookup table called dp of n rows and 2∗sum(arr)+1 columns. The number of rows represents the number of given integers and the number of columns represents all possible target sums that can be built using these integers. For any given integer, whenever we generate and evaluate an expression, we store it at dp[i][T], where i and T represent the index of the integer and the generated sum, respectively. 
O(n * k), k = sum(arr) - O(n * k)
Bottom-up solution:
create a lookup table of size n×2∗sum(arr)+1. The table is initialized with 0 except for the indexes corresponding to the first integer. Two expressions can be generated using the first integer by inserting a + or a − before it. Therefore, we store 1 at the respective indices.
the algorithm iterates over all integers and for every possible target sum t, it checks if it was generated during the previous iterations i.e., if dp[i-1][total+t] > 0, where total is the sum of all elements of the array. If it was generated, two new expressions are considered by inserting a + or a − before the current integer and the count of expressions is increased i.e., dp[i-1][total+t] is added to the values of dp[i][total+t+arr[i]] and dp[i][total+t-arr[i]].
O(n * k) - O(n * k)
'''
# Naive
def find_target_sum_ways(arr, T):
    return find_target_sum_ways_rec(arr, 0, T)

def find_target_sum_ways_rec(arr, i, T):
    # If all integers are processed
    if i == len(arr):
        # If target is reached
        if T == 0:
            return 1
        # If target is not reached
        return 0
 
    # Return total count of the following cases:
    #       1. Add current element to the target
    #       2. Subtract current element from the target
    return (find_target_sum_ways_rec(arr, i + 1, T + arr[i]) +
            find_target_sum_ways_rec(arr, i + 1, T - arr[i]))

# Optimized
# -- Top down
def find_target_sum_ways(arr, T):
    total = sum(arr)

    # If the target can't be generated using the given numbers
    if total < abs(T):
        return 0
    
    # Initialize a lookup table
    dp = [[-1 for _ in range(2*total+1)] for _ in range(len(arr))]
    
    return find_target_sum_ways_rec(arr, 0, T, dp)

def find_target_sum_ways_rec(arr, i, T, dp):
    # If all integers are processed
    if i == len(arr):
        # If target is reached
        if T == 0:
            return 1
        # If target is not reached
        return 0

    #If we have solved it earlier, then return the result from memory
    if dp[i][T] != -1:
        return dp[i][T]
    
    # Calculate both sub-problems and save the results in the memory
    dp[i][T] = find_target_sum_ways_rec(arr, i + 1, T + arr[i], dp) + \
               find_target_sum_ways_rec(arr, i + 1, T - arr[i], dp)
    
    return dp[i][T]

# -- Bottom up
def find_target_sum_ways(arr, T):
    total = sum(arr)

    # If the target can't be generated using the given numbers
    if total < abs(T):
        return 0    

    # Initialize a lookup table
    dp = [[0 for _ in range(2*total+1)] for _ in range(len(arr))]
    dp[0][total + arr[0]] = 1
    dp[0][total - arr[0]] += 1

    # For every integer
    for i in range(1, len(arr)):
        # For every possible target sum
        for t in range(-total, total+1):
            # If at least one expression (during previous iterations) evaluated to this target sum
            if dp[i - 1][total + t] > 0:
                dp[i][total + t + arr[i]] += dp[i - 1][total + t]
                dp[i][total + t - arr[i]] += dp[i - 1][total + t]
    
    return dp[len(arr)-1][T+total]

## Subset Sum
'''
Given a set of positive numbers arr and a value total, determine if there exists a subset in the given set whose sum is equal to total. A subset can be an empty set, or it can either consist of some elements of the set or all the elements of the set.
Naive approach: all the possible combinations from the set, which makes our total. The recursive solution will be to check every element of the arr:
If it does not contribute to making up the total, we ignore that number and proceed with the rest of the elements, i.e., subset_sum_rec(arr, n-1, total).
OR
If it does contribute to making up the total, we subtract the total from the current number and proceed with the rest of the elements till the given total is zero, i.e., subset_sum_rec(arr, n-1, total-arr[n-1]).
O(2^n) - O(n)
---
Optimized solution
Top-down solution: we will use a 2-D table dp that will store the computed results at each step. Whenever we encounter a subproblem whose value is already calculated, we will simply look up from the table instead of recalculating it. The last index of dp will contain the required output.
O(n * m) - O(n * m)
Bottom-up solution: A 2-D table of size [n + 1] * [total + 1] is used here. We will initialize the table so that the rows will represent the possible subsets, and the columns will show the total we need to achieve. At any given time, each column represents the amount that we have to calculate using the elements of the given set at the respective row. Now, there will be a check for each element as to whether it will contribute to making up the total or not.
O(n * m) - O(n * m)
'''
# Naive
def subset_sum(arr, total):
    n = len(arr)
    return subset_sum_rec(arr, n, total)

# helper function
def subset_sum_rec(arr, n, total): 
    # base case
    if total == 0:
        return True
    
    if n == 0:
        return False

    if (arr[n-1] > total):
        return subset_sum_rec(arr, n-1, total)

    # We either exclude the element or include the element
    return subset_sum_rec(arr, n-1, total) or subset_sum_rec(arr, n-1, total-arr[n-1])

# Optimized
# -- Top down
def subset_sum(arr, total): # main function
    n = len(arr)
    dp = [[-1 for i in range(total + 1)] for j in range(n + 1)]
    return subset_sum_rec(arr, n, total, dp)


def subset_sum_rec(arr, n, total, dp): # helper function
    # Base case
    if total == 0:
        return True
    
    if n == 0:
        return False

    # If we have solved it earlier, then return the result from memory
    if dp[n][total] != -1:
        return dp[n][total]

    #Otherwise, we calculate it and store it for later use
    if (arr[n-1] > total):
        dp[n][total] = subset_sum_rec(arr, n-1, total, dp)
        return dp[n][total]
    
    # We either exclude the element or include the element
    dp[n][total] = subset_sum_rec(arr, n-1, total, dp) or subset_sum_rec(arr, n-1, total-arr[n-1], dp)
    return dp[n][total]

# -- Bottom up
def subset_sum(arr, total): # main function
    n = len(arr)
    dp = [[False for i in range(total + 1)] for j in range(n + 1)]

    # Bases cases
    for i in range (n):
        for j in range(total):
            if i == 0:
                dp[i][j] = False

            if j == 0:
                dp[i][j] = True

    # solving for the total 
    for i in range(1, n + 1):
        for j in range(1, total + 1):
            # if last element is greater than total we exclude it
            if arr[i-1] > j:
                dp[i][j] = dp[i-1][j]
            else:
                # otherwise we proceed on to rest of the elements
                # we either exclude the element or include the element
                dp[i][j] = dp[i-1][j] or dp[i-1][j-arr[i-1]]
    
    return dp[n][total]

## Count of Subset Sum
'''
Given a set of positive numbers nums and a value target_sum, count the total number of subsets of the given set whose sum is equal to the target_sum.
Naive approach: we will divide our problem into smaller subproblems, starting from the start of the nums list and for each element, we will do the following steps:
Check for the following base cases:
- If the target_sum is 0, it means that we can reach the sum of 0 without including any element in our subset resulting in an empty subset, so we return 1 to include an empty subset.
- If we have traversed all of our elements then we cannot proceed further to calculate the required sum. So we return 0.
Consider the current element for the required sum.
- If it contributes to making up the target_sum, we subtract the target_sum from the current number and proceed on to the next number with the new target sum of target_sum – nums[current_index].
- Do not consider the contribution of the current element for the target_sum and move on to consider the rest of the numbers.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: The 2-D array will be of size input set size×(target+1) and initialized with −1 to indicate that these subproblems are yet to be solved. The rows correspond to each element in the input array and the columns correspond to the remaining sums from 0 to the target sum. So, for a given number and the remaining sum, we’ll do the following in the helper function:
Check the base cases as done in the naive approach.
If you look at the code shown below, you will see that we are checking if the count of subsets for a given element and the remaining sum has already been evaluated or not.
- If it hasn’t already been evaluated, evaluate it as done in the naive approach and store the result in the 2-D array at lookup_table[current_index][required_sum]. The current_index specifies the index of the current element in the nums array.
- If it has already been evaluated, fetch the computed result from lookup_table[current_index][required_sum].
O(n * s) - O(n * s)
Bottom-up solution: A 2-D array of size s×(t+1), where s is the input set size and t is the target sum, which is used to store the results of subproblems. All the cells in the table are initialized to 0. In each iteration, we consider a number from the input set and check if that number can be included in our subset to reach the remaining sum. If you look at the code shown below, you will see the following steps to calculate the count of the subset sum:
Check the base case for the first number:
- If the first number in the input array is 0, we update the cell at lookup[0][0] to 2 because the empty set and the set 0 will both contribute to the zero sum.
- If the first number is not 0, we update the cell at lookup[0][0] to 1 because only the empty set will contribute to the zero sum. Also, if the number is less than the target, it’ll contribute towards the target, so update the cell at lookup[0][nums[0]] to 1.
We iterate the input array starting from the 1st index, and for each of the numbers, we perform two steps for each required_sum from 0 to target:
- Exclude the number. Get the count of all the subsets before the given number by fetching the value from lookup[current-1][required_sum].
- Include the number if its value is not more than the required_sum. Get the count of all the subsets for the remaining target of required_sum - nums[current] by fetching it from lookup[current-1][required_sum - nums[current]].
- Add the counts and store it in lookup[current][required_sum].
After all the calculations, lookup[nums_len - 1][target_sum] has the count of all the subsets having the sum equal to the target_sum.
O(n * s) - O(n * s)
'''
# Naive 
# main function
def count_subset_sum(nums, target_sum):
    return count_subset_sum_rec(nums, target_sum, 0)

# helper function
def count_subset_sum_rec(nums, target_sum, current_index):
    # base cases
    if target_sum == 0:
        return 1
    if current_index >= len(nums):
        return 0

    # if the number at current_index does not exceed the 
    # target_sum, we include it
    sum1 = 0
    if nums[current_index] <= target_sum:
        sum1 = count_subset_sum_rec(nums, target_sum - nums[current_index], current_index + 1)

    # recursive call, excluding the number at the current_index
    sum2 = count_subset_sum_rec(nums, target_sum, current_index + 1)

    return sum1 + sum2

# Optimized
# -- Top down
def count_subset_sum(nums, target_sum):
    # lookup table of size len(nums) x (targetSum + 1)
    lookup_table = [[-1 for i in range(target_sum + 1)] for j in range(len(nums))]
    return count_subset_sum_rec(nums, target_sum, 0, lookup_table)

# helper function
def count_subset_sum_rec(nums, required_sum, current_index, lookup_table):
    # base cases
    if required_sum == 0:
        return 1 
    if current_index >= len(nums):
        return 0    

    # check if we have processed the same subproblem already or not
    if lookup_table[current_index][required_sum] == -1:
        # if the number at current_index does not exceed the 
        # required_sum, we include it
        sum1 = 0
        if nums[current_index] <= required_sum:
            sum1 = count_subset_sum_rec(nums, required_sum - nums[current_index], current_index + 1, lookup_table)

        # recursive call, excluding the number at the current_index
        sum2 = count_subset_sum_rec(nums, required_sum, current_index + 1, lookup_table)

        lookup_table[current_index][required_sum] = sum1 + sum2

    return lookup_table[current_index][required_sum]

# -- Bottom up
def count_subset_sum(nums, target_sum):
    nums_len = len(nums)
    # lookup table of size nums_len x (targetSum + 1)
    lookup = [[0 for i in range(target_sum + 1)] for j in range(nums_len)]

    # Base case 1
    if nums[0] == 0:
        lookup[0][0] = 2
    # Base case 2
    else: 
        lookup[0][0] = 1
        if nums[0] <= target_sum:
            lookup[0][nums[0]] = 1

    for current in range(1, nums_len):
        for required_sum in range(0, target_sum + 1):
            # included
            sum1 = 0
            if nums[current] <= required_sum:
                sum1 = lookup[current - 1][required_sum - nums[current]]
                
            # excluded
            sum2 = lookup[current - 1][required_sum]

            lookup[current][required_sum] = sum1 + sum2
    
    return lookup[nums_len - 1][target_sum]