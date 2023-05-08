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

## Partition Array Into Two Arrays to Minimize Sum Difference
'''
Suppose you are given an array, nums, containing positive numbers. You need to partition the array into two arrays such that the absolute difference between their sums is minimized.
Naive approach: we divide the problem into subproblems, and for each array element, we decide whether to place it in the first or second partitioned array. This is done using the following rules:
Base case: If we reach the end of the array, and there are no elements to add in either of the partitioned arrays, we return the absolute difference between the sums of the two arrays.
Otherwise, we calculate the difference in the sums of the two arrays for the two scenarios:
- add the current element to the first partitioned array
- add it to the second partitioned array
We then select the option that results in the minimum difference.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: n the recursive approach, the following three variables kept changing:
- The array index, i.
- The sum of the first partitioned array, sum1.
- The sum of the second partitioned array, sum2.
We will use a 2-D table, dp with i rows, and sum1 + 1 columns to store the result, i.e., the difference between the sums of the partitioned arrays. We haven’t considered sum2 since, for a given i and sum1, sum2 of the remaining numbers will always be the same.
O(n * S) - O(n * S)
Bottom-up solution: Check solution
'''
# Naive
# Helper function with updated signature: i is current index in nums
# sums1 is the sum of the first partitioned array
# sums2 is the sum of the second partitioned array
def minimum_partition_array_sum_difference_helper(nums, i, sum1, sum2):

    # Base case: If i becomes equal to the length of nums, there are no more
    # elements left to add, so return the absolute difference between the
    # two sums
    if i == len(nums):
        return abs(sum1 - sum2)

    # Otherwise, recuresively calculate the minimum of adding the current
    # array element to either the first, or second partitioned array
    return min(
        minimum_partition_array_sum_difference_helper(nums, i + 1, sum1 + nums[i], sum2),
        minimum_partition_array_sum_difference_helper(nums, i + 1, sum1, sum2 + nums[i]),
    )

def minimum_partition_array_sum_difference(nums):
    return minimum_partition_array_sum_difference_helper(nums, 0, 0, 0)

# Optimized
# -- Top down
# Helper function with updated signature: i is current index in nums
# sums1 is the sum of the first partitioned array
# sums2 is the sum of the second partitioned array
def minimum_partition_array_sum_difference_helper(nums, i, sum1, sum2, dp):
    # Base case: If i becomes equal to the length of nums, there are no more
    # elements left to add, so return the absolute difference between the
    # two sums
    if i == len(nums):
        return abs(sum1 - sum2)

    # If the 2-D array contains the default value of -1, update its value with the minimum of adding the current
    # array element to either the first, or second partitioned array
    if dp[i][sum1] == -1:

        dp[i][sum1] = min(
            minimum_partition_array_sum_difference_helper(
                nums, i + 1, sum1 + nums[i], sum2, dp
            ),
            minimum_partition_array_sum_difference_helper(
                nums, i + 1, sum1, sum2 + nums[i], dp
            ),
        )

    # Return the value stored in the 2-D array if the sub-problem has already been computed
    return dp[i][sum1]

def minimum_partition_array_sum_difference(nums):
    # Initializing the 2-D array
    dp = [[-1 for x in range(sum(nums) + 1)] for y in range(len(nums))]

    return minimum_partition_array_sum_difference_helper(nums, 0, 0, 0, dp)

# -- Bottom up
def minimum_partition_array_sum_difference(nums):

    # Calculating the sum of the original array
    sum_array = sum(nums)

    # Calculating the number of rows and columns in the 2-D array
    rows = len(nums)
    cols = (sum_array // 2) + 1

    # Initializing the 2-D array
    dp = [[-1 for x in range(cols)] for y in range(rows)]

    # The first column will be initialized to all 1s, since a sum s = 0
    # will always be true if no elements are added to the subset
    for i in range(rows):
        dp[i][0] = 1

    # For the first row, each entry will be 1 if the sum s is equal to the
    # first element, and 0 otherwise
    for s in range(1, cols):
        dp[0][s] = nums[0] == s

    # Iterating and filling the dp array
    for i in range(1, rows):
        for s in range(1, cols):
            # Check if sum s can be obtained without nums[i] in the array
            if dp[i - 1][s]:
                dp[i][s] = dp[i - 1][s]

            # Check if sum s can be obtained with nums[i] in the array
            elif s >= nums[i]:
                dp[i][s] = dp[i - 1][s - nums[i]]

            # If neither of the above two conditions is true, sum s can not be
            # obtained with nums[i] included in the array
            else:
                dp[i][s] = 0

    # Find the largest index in the last row which is 1 and return the absolute
    # difference between the two sums
    for s in range(cols - 1, -1, -1):
        if dp[rows - 1][s] == 1:
            sum1 = s
            sum2 = sum_array - sum1
            return abs(sum2 - sum1)


## Minimum Number of Refueling Stops
'''
You need to find the minimum number of refueling stops that a car needs to make to cover a distance, target. For simplicity, assume that the car has to travel from west to east in a straight line. There are various fuel stations on the way, that are represented as a 2-D array of stations, i.e., stations[i] = [di,fi], where di is the distance in miles of the ith gas station from the starting position, and fi is the amount of fuel in liters that it stores. Initially, the car starts with k liters of fuel. The car consumes one liter of fuel for every mile traveled. Upon reaching a gas station, the car can stop and refuel using all the petrol stored at the station. In case it cannot reach the target, the program simply returns −1.
Naive approach: O(n * 2^n) - O(n)
Optimized solution: O(n^2) - O(n^2)
'''
#Naive
import math

# This function finds the maximum distance that can be travelled 
# by making "used" refuelling stops, considering fuel stations from index "index" onwards.
def min_refuel_stops_helper(index, used, cur_fuel, stations):
    # If no refuelling stops are made, return the current fuel level.
    if used == 0:
        return cur_fuel

    # If more refuelling stops are made than the number of fuel stations remaining,
    # return -inf (impossible to reach the target distance).
    if used > index:
        return -math.inf

    # Consider two options:
    # 1. Don't make a refuelling stop at the current fuel station.
    result1 = min_refuel_stops_helper(index - 1, used, cur_fuel, stations)

    # 2. Make a refuelling stop at the current fuel station.
    result2 = min_refuel_stops_helper(index - 1, used - 1, cur_fuel, stations)
    
    # Return the maximum of the two options, but if the fuel at the current fuel station
    # is not enough to reach the next fuel station, return -inf (impossible to reach the target distance).
    result = max(result1, -math.inf if result2 < stations[index - 1][0] else result2 + stations[index - 1][1])
    return result

# This function finds the minimum number of refuelling stops needed 
# to reach the target distance, given a starting fuel level and a list of fuel stations.
def min_refuel_stops(target, start_fuel, stations):
        
    n = len(stations)
    i = 0
    # Initialize an array to store the maximum distance that can be travelled
    # for each number of refuelling stops.
    max_d = [-1 for i in range(n+1)]
    # Find the maximum distance that can be travelled for each number of refuelling stops.
    while i <= n:
        max_d[i] = min_refuel_stops_helper(n, i, start_fuel, stations);
        i += 1
    result = -1
    i = 0
    # Find the minimum number of refuelling stops needed by iterating over max_d
    # and finding the first value that is greater than or equal to the target distance.
    while i <= n:
        if max_d[i] >= target:
            result = i
            break
        i += 1
    return result

# Optimized
# -- Top down
# This function finds the maximum distance that can be travelled 
# by making "used" refuelling stops, considering fuel stations from index "index" onwards.
def min_refuel_stops_helper(index, used, cur_fuel, stations, memo):
    # If no refuelling stops are made, memoize and return the current fuel level.
    if used == 0:
        memo[index][used] = cur_fuel
        return memo[index][used]

    # If more refuelling stops are made than the number of fuel stations remaining,
    # memoize and return -inf (impossible to reach the target distance).
    if used > index:
        memo[index][used] = -math.inf
        return memo[index][used]

    # if the solution already exists in the memo
    # return the result of the solution from memo
    if memo[index][used] != -1:
        return memo[index][used]

    # Consider two options:
    # 1. Don't make a refuelling stop at the current fuel station.
    result1 = min_refuel_stops_helper(index - 1, used, cur_fuel, stations, memo)

    # Make a refuelling stop at the current fuel station
    result2 = min_refuel_stops_helper(index - 1, used - 1, cur_fuel, stations, memo)
    
    # Memoize and return the maximum of the two options, but if the fuel at the current fuel station
    # is not enough to reach the next fuel station, return -inf (impossible to reach the target distance).
    memo[index][used] = max(result1, -math.inf if result2 < stations[index - 1][0] else result2 + stations[index - 1][1])
    return memo[index][used]
    
# This function finds the minimum number of refuelling stops needed 
# to reach the target distance, given a starting fuel level and a list of fuel stations.
def min_refuel_stops(target, start_fuel, stations):
        
    n = len(stations)
    # Initialize an array to store the maximum distance that can be travelled
    # for each number of refuelling stops.
    memo = [[-1 for i in range(n+1)] for j in range(n+1)]
    i = 0
    # Find the maximum distance that can be travelled for each number of refuelling stops.
    while i <= n:
        min_refuel_stops_helper(n, i, start_fuel, stations, memo);
        i += 1
    result = -1
    i = 0
    # Find the minimum number of refuelling stops needed by iterating over memo
    # and finding the first value that is greater than or equal to the target distance.
    while i <= n:
        if memo[n][i] >= target:
            result = i
            break
        i += 1
    return result
  
## -- Bottom up
def min_refuel_stops(target, start_fuel, stations):
        n = len(stations)
        # creating an array to store the maximum distances
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        i = 0
        # fill up the first column of the table with the start fuel.
        while (i <= n):
            dp[i][0] = start_fuel
            i += 1
        i = 1
        # iterating over all the stations from i = 1 to n
        while (i <= n):
            j = 1
            # checking fueling stops from j = 1 to j = i
            while (j <= i):
                # refuel at current station
                if (dp[i - 1][j - 1] >= stations[i - 1][0]):
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + stations[i - 1][1])
                # not refuel at current station
                else:
                    dp[i][j] = dp[i - 1][j]
                j += 1
            i += 1
        i = 0
        # After visiting all the stations, find minimum `j`
        while (i <= n):
            if (dp[n][i] >= target) :
                return i
            i += 1
        return -1


## Equal Sum Subarrays
'''
For a given array nums, determine if the array can be divided into two subarrays such that the sum of both the subarrays is equal.
Naive approach: We can solve this problem with the following two steps:
First, we calculate the sum of the array. If the sum of the array is odd, there can’t be two subarrays with an equal sum. Therefore, we return FALSE.
If the sum is even, we calculate Target Sum = Array Sum/2 and find a subarray of the array with a sum equal to Target Sum. The subarray can be found by either including the current element or not including it. To include the current element, we need to subtract it from the Target Sum.
O(2^n) - O(n)
Optimized solution
Top-down solution: Using memoization and recursion to go further for any element in the array, we can:
- Choose that element, in which case, the Target Sum will be updated as Target Sum = Target Sum − Nums [i].
- If we ignore the ith element and move forward, Target Sum will remain the same.
- If Target Sum == 0, then return TRUE.
- If Target Sum < 0, then return FALSE.
O(n * s) - O(n * s)
Bottom-up solution: Check solution
'''
# Naive
def can_partition_array(nums):
    nums_len = len(nums)
    # base case
    if nums_len == 1:
        return False
    # calculate sum of array
    array_sum = sum(nums)
    # if array_sum is odd, it cannot be partitioned into equal sum subarrays
    if array_sum % 2 != 0:
        return False

    # calculating the target subarray sum
    target_sum = array_sum // 2
    return is_subarray_sum(nums, nums_len - 1, target_sum)


def is_subarray_sum(nums, nums_len, target_sum):
    if target_sum == 0:  # subarray formed with required half-sum
        return True
    if nums_len == 0 or target_sum < 0:
        return False

    # here we will perform two operations:
    # 1. include the current element therefore 'target_sum'
    #    will be updated as 'target_sum - current element'
    # or
    # 2. exclude current element therefore no need to update 'target_sum'
    result = (is_subarray_sum
             (nums, nums_len - 1, target_sum - nums[nums_len - 1]) 
                or is_subarray_sum(nums, nums_len - 1, target_sum))  
    return result

# Optimized
# -- Top down
def can_partition_array(nums):
    nums_len = len(nums)
    # base case
    if nums_len == 1:
        return False

    # calculate sum of array
    array_sum = sum(nums)
    # if array_sum is odd, it cannot be partitioned into equal sum subarrays
    if array_sum % 2 != 0:
        return False

    # calculating the target subarray sum
    target_sum = array_sum // 2

    # Creating a memo to store two things:
    # (i) The index of array elements
    # (ii) The target sum
    memo = {}

    def is_subarray_sum(nums, index, target_sum):
        nums_len = len(nums)

        if target_sum == 0:  # Subarray formed with required half-sum
            return True
        if target_sum < 0 or index == nums_len:
            return False

        # here we will perform two operations:
        # 1. include the current element therefore target_sum
        #    will be updated as target_sum - current element
        # or
        # 2. exclude current element therefore no need to update target_sum
        if (index, target_sum) not in memo:
            memo[index, target_sum] = is_subarray_sum(
                nums, index + 1, target_sum - nums[index]) \
                    or is_subarray_sum(nums, index + 1, target_sum)


        return memo[index, target_sum]
    return is_subarray_sum(nums, 0, target_sum)
  
  # -- Bottom up
def can_partition_array(nums):
    array_sum = sum(nums)

    # if 'array_sum' is an odd number, we can't have 
    # two subarrays with equal sum
    if array_sum % 2 != 0:
        return False
    else:
    # We are trying to find a subarray of given numbers 
    # that has a total sum of 's/2'.
        target_sum = int(array_sum / 2)

    nums_len = len(nums)
    
    # Making a 2-D table.
    dp = [[0 for x in range(target_sum + 1)] for y in range(nums_len + 1)]
    
    # Intializng the first row with False and first column with True.
    for i in range(nums_len + 1):
        for j in range(target_sum + 1):
            if i == 0 and j == 0:
                dp[i][j] = True

            elif j == 0:
                dp[i][j] = True

            elif i == 0:
                dp[i][j] = False
    
    # Process all subarrays for all sums
    for i in range(1, nums_len + 1):
        for j in range(1, target_sum + 1):
            # if we can find a subset to get the remaining sum
            if nums[i - 1] <= j:
                dp[i][j] = dp[i - 1][j - nums[i - 1]] or dp[i - 1][j]
            # else we can get the sum 'j' without the number at index 'i'
            else:
                dp[i][j] = dp[i - 1][j]

    # Return the answer
    return dp[nums_len][target_sum]

## Count Square Submatrices
'''
Given a matrix containing only ones and zeros, count the total number of square submatrices that contain only ones.
If the matrix is empty, then return 0.
Naive approach: We iterate over the entire matrix and at each cell, we check if the value is 0 or 1. If the value is 1, we call the helper function which returns the count of square submatrices with all ones starting from the current cell. In the helper function, we perform the following tasks:
If the value of the current cell is 1, the helper function is called recursively for its right, bottom and bottom-right cells. We then take the minimum of the returned values and add 1. Taking the minimum value ensures that we are only considering square submatrices and 1 is added to count the cell itself as it is also a 1×1 square submatrix.
If the value of the current cell is 0, we return 0 as there is no possible square submatrice with all ones starting from this cell. Similarly, if the indices are out range, we return 0.
O(m * n * 3^mn)
Optimised solution
Top-down solution: O(m * n) - O(m * n)
Bottom-up solution: We create a lookup table of size m×n and copy the first row and column of the input matrix to the lookup table. We iterate over the remaining matrix, starting from matrix[1][1], and find the number of squares ending at the current cell and store them in the lookup table. To find the number of squares ending at the current cell, we can use the following recurrence relation:
- If matrix[i][j] == 0, then there is no possible square so we skip this iteration.
- If matrix[i][j] == 1, then we compare the size of the squares lookup_table[i-1][j-1], lookup_table[i-1][j], and lookup_table[i][j-1] and take the minimum of all three values and then, add 1 to it.
Finally, we calculate the sum of the lookup table which is equal to the number of square submatrices with all 1’s.
O(m * n) - O(m * n)
'''
# Naive
def count_squares(matrix):
    m = len(matrix)
    n = len(matrix[0])
    res = 0

    # iterate over the entire matrix
    for i in range(m):
        for j in range(n):
            # if the value of the cell is 1, call the helper function
            if matrix[i][j] == 1:
                res += count_squares_rec(matrix, i, j, m, n)
    
    return res

# helper function
def count_squares_rec(matrix, i, j, m, n):
    # if the indices are out of range or the value of the cell is 0, return 0
    if i >= m or j >= n or matrix[i][j] == 0:
        return 0
    
    # call the function recursively for the right, bottom and bottom right cells
    right = count_squares_rec(matrix, i, j+1, m, n)
    bottom = count_squares_rec(matrix, i+1, j, m, n)
    bottom_right = count_squares_rec(matrix, i+1, j+1, m, n)

    # return 1 plus the minimum of the three recursive calls
    return 1 + min(right, bottom, bottom_right)

# Optimized 
# -- Top down
def count_squares(matrix):
    m = len(matrix)
    n = len(matrix[0])

    # lookup table to store the results of our subproblems
    lookup_table = lookup_table = [[-1 for x in range(n)] for x in range(m)]
    res = 0

    # iterate over the entire matrix
    for i in range(m):
        for j in range(n):
            # if the value of the cell is 1 and the lookup table does not have
            # the result stored alreayd, call the helper function
            if matrix[i][j] == 1 and lookup_table[i][j] == -1:
                res += count_squares_rec(matrix, i, j, m, n, lookup_table)
            elif lookup_table[i][j] != -1:
                res += lookup_table[i][j]
    
    return res

# helper function
def count_squares_rec(matrix, i, j, m, n, lookup_table):
    # if the indices are out of range or the value of the cell is 0, return 0
    if i >= m or j >= n or matrix[i][j] == 0:
        return 0
    
    # call the function recursively for the right, bottom and bottom right cells
    # if the result is not already stored in the lookup table
    if lookup_table[i][j] == -1:
        lookup_table[i][j] = 1 + min(count_squares_rec(matrix, i, j+1, m, n, lookup_table), count_squares_rec(matrix, i+1, j, m, n, lookup_table), count_squares_rec(matrix, i+1, j+1, m, n, lookup_table))
    
    return lookup_table[i][j]

# -- Bottom up
def count_squares(matrix):
    # if the matrix is empty, return 0
    if len(matrix) == 0 or len(matrix[0]) == 0:
      return 0

    # create lookup table to store number of squares
    lookup_table = [[0 for x in range(len(matrix[0]))] for x in range(len(matrix))]
    result = 0

    # copy first row and column of input matrix to lookup table
    for i in range(len(matrix)):
        lookup_table[i][0] = matrix[i][0]
    for i in range(len(matrix[0])):
        lookup_table[0][i] = matrix[0][i]

    # iterate over the matrix and store the count od squares in lookup_table
    for i in range(1, len(matrix)):
      for j in range(1, len(matrix[0])):
        # If matrix[i][j] is equal to 0
        if (matrix[i][j] == 0):
          continue

        # there is at least one square submatrix at this location, hence the + 1
        # in addition, find the minimum number of square submatrices 
        # whose bottom-right corner is one of the neighbours of this location.
        lookup_table[i][j] = 1 + min(lookup_table[i - 1][j], lookup_table[i][j - 1], lookup_table[i - 1][j - 1])

    # sum up the values in the lookup_table to get the count of square submatrices
    for i in range(0, len(lookup_table)):
      for j in range(0, len(lookup_table[0])):
        result += lookup_table[i][j]

    return result

### *** Unbounded Knapsack
## Unbounded Knapsack 
'''
Suppose you have a list of weights and corresponding values for n items. Each item will have a weight and a certain value associated with it. You have a knapsack that can carry items up to a specific maximum weight, known as the capacity of the knapsack.
You want to maximize the sum of values of the items in your knapsack while ensuring that the sum of the weights of the items remains less than or equal to the knapsack’s capacity. If all the combinations exceed the given knapsack’s capacity, return 0.
Naive approach: O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n * W) - O(n * W)
Bottom-up solution:  O(n * W) - O(n * W)
'''
# Naive
def unbounded_knapsack_rec(weights, values, n, capacity):
    # Base case
    if(n == 0):
        return (capacity//weights[0]) * values[0]
    
    # Check if the weight of the nth item is less than capacity 
    # If it is, we have two choices
    # 1) Include the item 
    # 2) Don't include the item
    if(weights[n] <= capacity):
        taken = values[n] + unbounded_knapsack_rec(weights,values,n,capacity-weights[n])
        not_taken = 0 + unbounded_knapsack_rec(weights, values, n-1, capacity)
        
        # As we want to maximize the profit, we take maximum of the two options
        return max(taken, not_taken)
    
    # If weight of the nth item is greater than the capacity
    # Don't include the item
    else:
        return unbounded_knapsack_rec(weights, values, n-1, capacity)
    
def unbounded_knapsack(weights, values, n, capacity):
    return unbounded_knapsack_rec(weights, values, n-1, capacity)

# Optimized
# -- Top down
def unbounded_knapsack_rec(weights, values, n, capacity, dp):
    # Base case
    if(n == 0):
        return (capacity//weights[0]) * values[0]
    
    # If we have solved it earlier, then return the result from memory
    if dp[n][capacity] != -1:
        return dp[n][capacity]

    # Check if the weight of the nth item is less than capacity 
    # If it is, we have two choices
    # 1) Include the item 
    # 2) Don't include the item
    if(weights[n] <= capacity):
        taken = values[n] + unbounded_knapsack_rec(weights, values, n, capacity-weights[n], dp)
        not_taken = 0 + unbounded_knapsack_rec(weights, values, n-1, capacity, dp)
        # As we want to maximize the profit, we take maximum of the two values
        dp[n][capacity] = max(taken, not_taken)
    
    # If weight of the nth item is greater than the capacity
    # Don't include the item
    else:
        dp[n][capacity] = unbounded_knapsack_rec(weights, values, n-1, capacity, dp)
        
    return dp[n][capacity]
    
def unbounded_knapsack(weights, values, n, capacity):
    dp = [[-1 for i in range(capacity + 1)] for j in range(n + 1)]
    return unbounded_knapsack_rec(weights, values, n-1, capacity, dp)

# -- Bottom up
def unbounded_knapsack(weights, values, n, capacity):
    dp = [[0 for i in range(capacity + 1)] for j in range(n + 1)]
    
    # Base case
    for i in range(weights[0], capacity+1):
        dp[0][i] = (i//weights[0]) * values[0]
    
    for i in range(1,n):
        for j in range(0,capacity+1):

            # Check if the weight of the nth item is less than capacity 
            # If it is, we have two choices
            # 1) Include the item 
            # 2) Don't include the item
            if (weights[i] <= j):
                taken = values[i]+ dp[i][j-weights[i]]
                not_taken = 0 + dp[i-1][j]
                dp[i][j] = max(taken, not_taken)
                
            # If weight of the nth item is greater than the capacity
            # Don't include the item
            else:
                dp[i][j] = dp[i-1][j]
 
    return dp[n-1][capacity]

## Maximum Ribbon Cut
'''
Given a ribbon of length n and a set of possible sizes, cut the ribbon in sizes such that n is achieved with the maximum number of pieces.
You have to return the maximum number of pieces that can make up n by using any combination of the available sizes. If the n can’t be made up, return -1, and if n is 0, return 0.
Naive approach: O(2^n+k) - O(n + k)
---
Optimized solution
Top-down solution: We start our solution by creating a table dp and initializing it with -1 and a helper function that assists us in calculating the number of ribbons. It has three cases as follows:
If the remaining amount is equal to zero, we return 0.
If the length of list size has reached 0 or index reaches the size, we return -1.
Then we make two recursive calls for count_ribbon_pieces_helper. The first call is made only if the ribbon length at index does not exceed n by keeping index at the same position and subtracting the ribbon length at index from the total ribbon length. The returned value is incremented by 1 and stored in c1. The second recursive call is made by excluding the ribbon length by jumping index to the next position. The returned value is stored in c2. Every time, the maximum of c1 and c2 is stored in dp[index][n] so that it can be reused again next time.
O(n*k) - O(n*k)
Bottom-up solution: The idea is before calculating dp[i], we have to compute all maximum counts for ribbon sizes up to i. In each iteration i of the algorithm dp[i] is computed as dp[i]=max j=0...n−1 ​ (dp[i],dp[i−cj ​ ]+1)
O(n * k) - O(n)
'''
# Naive
def count_ribbon_pieces(n, sizes):
  maximum = count_ribbon_pieces_helper(sizes, n, 0)
  if maximum == -1:
    return -1
  else:
    return maximum


def count_ribbon_pieces_helper(sizes, n, index):
  # base case
  if n == 0:
    return 0

  length = len(sizes)
  # if the length is zero or the index is greater than or equal to the length,
  # then return -1 as the ribbon cannot be cut further.
  if length == 0 or index >= length:
    return -1  

  # recursive call after selecting the ribbon length at the index
  # if the ribbon length at the index exceeds the n, we shouldn't process this
  # since ribbon length is always positive, therefore initializing c1 with -1
  c1 = -1
  if sizes[index] <= n:
    maxSize = count_ribbon_pieces_helper(sizes, n - sizes[index], index)
    if maxSize != -1:
      c1 = maxSize + 1

  # recursive call after excluding the ribbon length at the curr
  c2 = count_ribbon_pieces_helper(sizes, n, index + 1)
  return max(c1, c2)

# Optimized
# -- Top down
def count_ribbon_pieces(n, sizes):
  length = len(sizes)
  # we created a table here
  dp = [[-1 for _ in range(n+1)] for _ in range(length)]

  result = count_ribbon_pieces_helper(sizes, n, 0, dp)
  
  if result == -1:
    return -1
  else:
    return result


def count_ribbon_pieces_helper(sizes, n, index, dp):
  # base case
  if n == 0:
    return 0

  length = len(sizes)
  # if the length is zero or the index is greater than or equal to the length,
  # then return -1 as the ribbon cannot be cut further.
  if length == 0 or index >= length:
    return -1

  if dp[index][n] == -1:
    # recursive call after selecting the ribbon length at the index
    # if the ribbon length at the index exceeds the n, we shouldn't process this
    # since ribbon length is always positive, therefore initializing c1 with -1
    c1 = -1
    if sizes[index] <= n:
      res = count_ribbon_pieces_helper(sizes, n-sizes[index], index, dp)
      if(res != -1):
        # recursive call after excluding the ribbon length at the index
        c1 = res + 1
    
    c2 = count_ribbon_pieces_helper(sizes, n, index+1, dp)
    dp[index][n] = max(c1, c2)

  return dp[index][n]

# -- Bottom up
def count_ribbon_pieces(n, sizes):
    # create the array to store the results
    dp = [-1]*(n+1)
    dp[0] = 0
    # calculate the results for all combinations
    # and select the maximum
    for i in range(1, n+1):
      for c in sizes:
        if i-c >= 0 and dp[i-c] != -1:
          dp[i] = max(dp[i], 1 + dp[i-c])
    
    if dp[n] != -1:
        return dp[n]
    else:
        return -1
    
## Rod Cutting
'''
You are given a rod of length n meters. You can cut the rod into smaller pieces, and each piece has a price based on its length. Your task is to earn the maximum revenue that can be obtained by cutting up the rod into smaller pieces.
Naive approach: We will divide our problem into smaller subproblems, starting from the start of the lengths list and for each length, we will do the following steps:
If the remaining rod length is zero or if we have traversed all of our lengths, we return 0. This represents the base case where no further cutting is possible.
Otherwise, perform the following two steps:
- If we can cut a piece of length lengths[curr], where curr specifies the index of the piece length we are considering from the lengths array, we’ll add its price into our earned revenue and recursively evaluate the maximum earnings from the remaining rod. Else, if we cannot cut a piece of length lengths[curr], we’ll simply move to the next step.
- Evaluate the maximum earning without cutting a piece of current length from the rod.
- Return the maximum earnings earned from both of the steps.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: we need a 2-D table of size len(lengths)×n + 1 to store the maximum revenue earned for each rod of length n into a pieces of given lengths. We start our solution by creating a table dp and initializing it with -1 and a helper function that assists us in calculating the maximum revenue earned. If you look at the code below, you’ll see that the helper function has the following steps:
- If the remaining rod length is zero or if we have traversed all of our lengths, we return 0.
- If we haven’t already computed a revenue for a given rod length n and a current length at lengths[curr], we compute it as we did in the naive approach and store the result at dp[curr][n]. Here, curr specifies the index of the piece length we are considering from the lengths array.
- Otherwise, we return the already computed result from the table by fetching it from dp[curr][n].
The value at dp[curr][n] represents the maximum earning that can be obtained by cutting a rod of length n into pieces of length lengths[curr]. 
O(n * k) - O(n * k)
Bottom-up solution: Check solution 
O(n * k) - O(n * k)
'''
# Naive
def rod_cutting(lengths, prices, n):
    if len(prices) == 0 or len(prices) != len(lengths):
        return 0
    return rod_cutting_rec(lengths, prices, n, 0)

def rod_cutting_rec(lengths, prices, n, curr):
    # base case
    if n == 0 or curr == len(lengths):
        return 0

    # Cut the piece of size length[curr] 
    revenue1 = 0
    if lengths[curr] <= n:
        revenue1 = prices[curr] + rod_cutting_rec(lengths, prices, n - lengths[curr], curr)

    # Don't cut the piece from the rod and move to the next available length
    revenue2 = rod_cutting_rec(lengths, prices, n, curr + 1)

    # return the maxiumum of both revenues
    return max(revenue1, revenue2)
# Optimized
# -- Top down
def rod_cutting(lengths, prices, n):
    # Base case
    if len(prices) == 0 or len(prices) != len(lengths):
        return 0

    # Creating a lookup table of size len(lengths) x (n + 1)
    dp = [[-1 for _ in range(n+1)] for _ in range(len(lengths))]
    return rod_cutting_rec(lengths, prices, n, 0, dp)

# Helper function
def rod_cutting_rec(lengths, prices, n, curr, dp):
    # base case
    if n == 0 or curr == len(lengths):
        return 0

    # If a piece of size lengths[curr] is not already computed 
    # for a rod of length n, compute it
    if dp[curr][n] == -1:
        # Cut the piece of size length[curr]
        revenue1 = 0
        if lengths[curr] <= n:
            revenue1 = prices[curr] + rod_cutting_rec(lengths, prices, n - lengths[curr], curr, dp)

        # Don't cut the piece from the rod and move to the next available length
        revenue2 = rod_cutting_rec(lengths, prices, n, curr + 1, dp)
        
        # store the max in the lookup table
        dp[curr][n] = max(revenue1, revenue2)

    # return from the lookup table
    return dp[curr][n]

# -- Bottom up
def rod_cutting(lengths, prices, n):
    lengthsCount = len(lengths)
    pricesCount = len(prices)
    
    # base cases
    if n == 0 or pricesCount == 0 or pricesCount != lengthsCount:
        return 0
    
    # Creating a lookup table of size len(lengths) x (n + 1)
    dp = [[0 for _ in range(n+1)] for _ in range(lengthsCount)]

    # process all rod lengths for all given lengths
    for curr in range(lengthsCount):
        for rod_length in range(1, n + 1):
            # Fetch the maximum revenue obtained by selling the rod
            # of size rod_length - lengths[curr]
            revenue1 = revenue2 = 0
            if lengths[curr] <= rod_length:
                revenue1 = prices[curr] + dp[curr][rod_length - lengths[curr]]
            
            # Fetch the maximum revenue obtained without cutting the rod
            if curr > 0:
                revenue2 = dp[curr - 1][rod_length]
            
            # store the result in the table
            dp[curr][rod_length] = max(revenue1, revenue2)

    # maximum revenue will be at the bottom-right corner.
    return dp[lengthsCount - 1][n]

## Minimum Coin Change
'''
You’re given an integer total and a list of integers called coins. The integers inside the coins represent the coin denominations, and total is the total amount of money.
You have to return the minimum number of coins that can make up the total amount by using any combination of the available coins. If the amount can’t be made up, return -1. If the total amount is 0, return 0.
Naive approach: recursively calling calculate_minimum_coins_rec and subtracting the sum, and reducing the total until it reaches zero. From these combinations, choose the one with the minimum number of coins and return that number. If the sum of any combinations is not equal to total then return -1.
O(m^n) - O(n)
---
Optimized solution
Top-down solution: We start our solution by creating a helper function that assists us in calculating the number of coins we need. It has three base cases to cover what to return if the remaining amount is:
- Less than zero
- Equal to zero
- Not either of the former two cases
In the last case, we traverse the coins array, and at each element, call the calculate_minimum_coins function, passing rem minus the value of the coin to it. We store the return value of the base cases in a variable named result. We then add 1 to the result variable and assign this value to minimum, which is initially set to infinity at the start of each path. At the end of each path traversal, we update the (rem−1)th index of the array list with minimum if it is not equal to infinity, otherwise -1. Finally, we return the value at that index.
O(m*n) - O(m)
Bottom-up solution:
O(m*n) - O(m)
'''
# Naive
def coin_change(coins, total): 
    if total < 1:
        return 0
    return calculate_minimum_coins_rec(coins, total)

def calculate_minimum_coins_rec(coins, rem):  
    # Helper function that calculates amount left to be 
    # calculated and tells what it's value can be.
    if rem < 0: 
        return -1
    if rem == 0:
        return 0
    minimum = math.inf

    for s in coins: 
        # Recursive approach to keep in account every number's result 
        result = calculate_minimum_coins_rec(coins, rem-s)
        if result >= 0 and result < minimum:
            minimum = 1 + result

    if minimum != math.inf:
        return minimum
    else:
        return -1
# Optimized
# -- Top down
def coin_change(coins, total):
    if total < 1:
        return 0
    return calculate_minimum_coins(coins, total, [math.inf] * total)

def calculate_minimum_coins(coins, rem, array):  
    # Helper function that calculates amount left to be calculated and 
    # tells what it's value can be.
    if rem < 0: 
        return -1
    if rem == 0:
        return 0
    if array[rem-1] != math.inf:
        return array[rem-1]
    minimum = math.inf

    for s in coins: 
        # Recursive approach to keep in account 
        # of every number's result 
        result = calculate_minimum_coins(coins, rem-s, array)
        if result >= 0 and result < minimum:
            minimum = 1 + result

    if minimum !=  math.inf:
        array[rem-1] =  minimum
        return array[rem-1]
    else:
        return -1

# -- Bottom up
def coin_change(coins, total):
    if total < 1:
        return 0
    return calculate_minimum_coins(coins, total)
    
def calculate_minimum_coins(coins, rem):  
    # Helper function that calculates amount left to be calculated 
    # and tells what its value can be.
    dp = [rem+1] * (rem+1)
    dp[0] = 0 

    for i in range(1, rem+1):
      for c in coins:
        if i-c >= 0:
          dp[i] = min(dp[i], 1 + dp[i-c])
    
    if dp[rem] != (rem+1):
        return dp[rem]
    else:
        return -1

## Coin Change II
'''
Suppose you are given a list of coins and a certain amount of money. Each coin in the list is unique and of a different denomination. You are required to count the number of ways the provided coins can sum up to represent the given amount. If there is no possible way to represent that amount, then return 0.
Naive approach: While making the combinations, a point to keep in mind is that we should try to avoid repeatedly counting the same combinations. For example, 10+20 cents add up to 30 cents, and so do 20+10. In the context of combinations, both these sequences are the same, and we will count them as one combination. We can achieve this by using a subset of coins every time we consider them for combinations. The first time, we can use all n coins, the second time, we can use n−1 coins, that is, by excluding the largest one, and so on. This way, it is not possible to consider the same denomination more than once.
As a base case, we return 1 when the target amount is zero because there is only one way to represent zero irrespective of the number and kinds of coins available. Similarly, for the second base case, if at any point during our search for combinations, the remaining value needed to reach the total amount becomes less than zero, we return 0.
O(n^c) - O(c)
---
Optimized solution
Top-down solution: We store all the results in memo and then retrieve them as needed. Since we had two defining variables here, the amount, and the maximum value, we use them to uniquely identify each subproblem.
O(cn) - O(cn)
Bottom-up solution: Check solution
O(cn) - O(cn)
'''
# Naive
def count_ways_rec(coins, amount, maximum):
  if amount == 0:     # base case 1
    return 1
  if amount < 0:      # base case 2
    return 0
  ways = 0

  # iterate over coins
  for coin in coins:    

    # to avoid repetition of similar sequences, use coins smaller than maximum
    if coin <= maximum and amount - coin >= 0:  
      
      # notice how maximum is set to the current value of coin in recursive call    
      ways += count_ways_rec(coins, amount-coin, coin)  
  return ways

def count_ways(coins, amount):
  return count_ways_rec(coins, amount, max(coins))
# Optimized
# -- Top down
def count_ways_rec(coins, amount, maximum, memo):
  if amount == 0:     # base case 1
    return 1
  if amount < 0:      # base case 2
    return 0
  if (amount, maximum) in memo: # checking if memoized
    return memo[(amount, maximum)]
  ways = 0
  
  # iterate over coins
  for coin in coins:     

    # to avoid repetition of similar sequences, use coins smaller than maximum
    if coin <= maximum:     
      
      # notice how maximum is set to the current value of coin in recursive call
      ways += count_ways_rec(coins, amount-coin, coin, memo)  
  memo[(amount, maximum)] = ways #memoizing
  return ways

def count_ways(coins, amount):
  memo = {}
  return count_ways_rec(coins, amount, max(coins), memo)
# -- Bottom up
def count_ways(coins, amount):
  if amount == 0:     # base case 1
    return 1
  if amount <= 0:     # base case 2
    return 0

  # create and initialize the 2-D array
  dp = [[1 for _ in range(len(coins))] for _ in range(amount + 1)]

  # iterate over the 2-D array and update the values accordingly
  for amt in range(1, amount+1):
    for j in range(len(coins)):

      # keep the count of solutions including coins[j]
      coin = coins[j]
      if amt - coin >= 0: 
        x = dp[amt - coin][j]
      else:
        x = 0
        
      # keep the count of solutions excluding coins[j]
      if j >= 1:
        y = dp[amt][j-1]
      else:
        y = 0
      dp[amt][j] = x + y
  return dp[amount][len(coins) - 1]

### *** Recursive Numbers
## Fibonacci Numbers
'''
Fibonacci numbers are a sequence of numbers where each number is the sum of the two preceding numbers. Your task is to find the nth Fibonacci number.
Naive approach: O(2^n) - O(n)
---
Optimized Solution
Top-down solution:O(n) - O(n)
Bottom-up solution: O(n) - O(n)
'''
# Naive
def get_fibonacci(n):
    # Base case for n = 0 and n = 1
    if n < 2:
        return n

    # Otherwise, calculate the Fibonacci number using the recurrence relation
    return get_fibonacci(n - 1) + get_fibonacci(n - 2)

# Optimized
# -- Top down
def get_fibonacci_memo(n, lookup_table):
    # Base case
    if n < 2:
        return n

    # Check if already present
    if lookup_table[n] != -1:
        return lookup_table[n]

    # Adding entry to table if not present
    lookup_table[n] = get_fibonacci_memo(n - 1, lookup_table) + \
                      get_fibonacci_memo(n - 2, lookup_table)
    return lookup_table[n]


def get_fibonacci(n):
    # Initializing the lookup table of size num + 1
    lookup_table = [-1] * (n + 1)
    return get_fibonacci_memo(n, lookup_table)

# -- Bottom up
def get_fibonacci(n):
    # Base case
    if n < 2:
        return n
    # Initializing the lookup table
    lookup_table = [0] * (n + 1)
    lookup_table[0] = 0
    lookup_table[1] = 1

    for i in range(2, n + 1):
        # Storing sum of the two preceding values
        lookup_table[i] = lookup_table[i - 1] + lookup_table[i - 2]

    return lookup_table[n]

## Staircase Problem
'''
A child is running up a staircase with n steps and can hop either 1 step, 2 steps, or 3 steps at a time. Your task is to count the number of possible ways that the child can climb up the stairs.
Naive approach: Given that you can only hop either 1, 2, or 3 steps at a time, we can say that the number of ways to climb n stairs is the sum of the following:
the number of ways to reach the (n−1)th stair, as we can then hop 1 step to reach the nth stair 
the number of ways to reach the (n−2)th stair, as we can then hop 2 steps to reach the nth stair 
the number of ways to reach the (n−3)th stair, as we can then hop 3 steps to reach the nth stair
We can now formulate our recursive algorithm to count the number of ways to climb n stairs:
Base case 1: If n<0, there are no more steps left to climb, so we return 0.
Base case 2: If n=0, we’re at the top of the staircase. As per the rules of the calculation, we return 1.
Recursive case: The total number of ways that the child can hop the stairs is:
countways(n - 1) + countways(n - 2) + countways(n - 3)
O(3^n) - O(n)
---
Optimized Solution
Top-down solution: O(n) - O(n)
Bottom-up solution: O(n) - O(n)
'''
# Naive
def count_ways(n):
    # Base Conditions
    if n < 0: 
        return 0

    elif n == 0:
        return 1

    # Check each possible combination
    else:
        return count_ways(n - 1) + count_ways(n - 2) + count_ways(n - 3)
# Optimized
# -- Top down
def count_ways_rec(n, lookup_table):
    # Negative staircases i.e., invalid input
    if n < 0:  
        return 0

    # If 0 staircases
    elif n == 0:  
        return 1

    # If already present in the table
    elif lookup_table[n] > -1:  
        return lookup_table[n]

    # If not present in the table
    else:   
        lookup_table[n] = count_ways_rec(n - 1, lookup_table) +\
                                         count_ways_rec(n - 2, lookup_table) +\
                                         count_ways_rec(n - 3, lookup_table)
    return lookup_table[n]


def count_ways(n):
    lookup_table = [-1 for x in range(n + 1)]
    return count_ways_rec(n, lookup_table)

# -- Bottom up
def count_ways(n):
    if n < 0:
      return 0
    if n==0:
        return 1

    # Initialize lookup table
    lookup_table = [0 for x in range(n + 1)]  

    # Setting the first three values
    lookup_table[0] = 1  
    lookup_table[1] = 1
    lookup_table[2] = 2

    for i in range(3, n + 1):
        # Fill up the table by summing up previous three values
        lookup_table[i] = lookup_table[i - 1] + lookup_table[i - 2] + lookup_table[i - 3]  
        
    # Return the nth Fibonacci number
    return lookup_table[n]  

## Number Factors
'''
Given a fixed list of numbers, [1,3,4], and a target number, n, count all the possible ways in which n can be expressed as the sum of the given numbers. If there is no possible way to represent n using the provided numbers, return 0.
Naive approach: To reach a target number n, we have the following three possibilities:
- Number of ways to reach n−1, as we can then add 1 to reach n
- Number of ways to reach n−3, as we can then add 3 to reach n
- Number of ways to reach n−4, as we can then add 4 to reach n
These are all valid ways to reach the target number n. We can now sum them up to get all the possible ways in which n can be expressed as a sum of the available numbers.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n) - O(n)
Bottom-up solution: We first create and initialize the fixed array, dp, that will store our results. Then, we set dp[0] to 1, which is our base case, that is, there is only one way to reach a target number of 0 by not choosing any of the available numbers.
To calculate the final result, we sum up the number of ways to get the target numbers n−1, n−3, and n−4. This is similar to our recurrence relation. 
During the calculations, if at any point, the remaining value needed to reach n becomes negative, we will simply return 0, as there is no way to reach a negative target number.
O(n) - O(n)
'''
# Naive
# Available numbers are 1, 3, and 4
def count_ways(n):

    # Setting up our base cases
    # We can not get a negative target number at any point, 
    # so we return 0 for negative values
    if n < 0:
        return 0

    # There is only 1 way to reach a target number of 0, 
    # by not using any available numbers
    if n == 0:
        return 1

    # Recursively calculate the number of ways using the
    # recurrence relation
    return count_ways(n - 1) + count_ways(n - 3) + count_ways(n - 4)
# Optimized
# -- Top down
# Available numbers are 1, 3, and 4
def count_ways_rec(n, memo):

    # Setting up our base cases
    # We can not get a negative target number at any point, 
    # so we return 0 for negative values
    if n < 0:
        return 0

    # There is only 1 way to reach a target number of 0, 
    # by not using any available numbers
    if n == 0:
        return 1

    if memo[n] == -1:
        # Recursively calculate the number of ways using the
        # recurrence relation and store in memo
        memo[n] = count_ways_rec(n - 1, memo) + count_ways_rec(n - 3, memo) + count_ways_rec(n - 4, memo)
    
    return memo[n]

def count_ways(n):

    # Initializing our solution array
    memo = [-1 for x in range(n+1)]

    # Set up the base case, 1 way to get to the number 0
    memo[0] = 1

    # Pass our array to the helper function
    return count_ways_rec(n, memo)

# -- Bottom up
# Available numbers are 1, 2, and 4
def count_ways(n):
    # Initializing our solution array
    dp = [0] * (n + 1)

    # Each array index holds the number of ways to
    # reach a number equal to the index
    dp[0] = 1

    # Variables to store sub-target numbers
    n1 = n3 = n4 = 0

    # Iteratively calculate the number of ways to reach a
    # target number and store it in the solutions' array
    for sn in range(1, n+1):
        # Return 0 if index is less than 0, otherwise
        # set to array value
        n1 = 0 if sn - 1 < 0 else dp[sn - 1]
        n3 = 0 if sn - 3 < 0 else dp[sn - 3]
        n4 = 0 if sn - 4 < 0 else dp[sn - 4]

        # Using our recurrence relation to calculate new answers
        dp[sn] = n1 + n3 + n4
    return dp[n]

## Count Ways to Score in a Game
'''
Suppose there is a game where a player can score either 1, 2, or 4 points in each turn. Given a total score, n, find all the possible ways in which you can score these n points.
Naive approach: to reach a total score of n, we have the following three possibilities:
- Number of ways to reach n−1, as we can then add 1 to reach n
- Number of ways to reach n−2, as we can then add 2 to reach n
- Number of ways to reach n−4, as we can then add 4 to reach n
These all are valid ways to reach a score of n. We can now sum them up to get all the possible ways to reach a total score of n.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n) - O(n)
Bottom-up solution: create and initialize the fixed array, dp, that will store our results. Then, we set dp[0] to 1, which is our base case, that is, there is only one way to reach a score of 0 by not playing any moves.
To calculate the result for any turn, we sum up the number of ways to get a score of n−1, n−2, and n−4. This is similar to our recurrence relation.
During the calculations, if at any point, the remaining value needed to reach n becomes negative, we will simply return 0, as there is no way to reach a negative score.
O(n) - O(n)
'''
# Naive
# Scoring options are 1, 2, and 4
def scoring_options(n):
    # Setting up our base cases

    # We can not get a negative score, we return 0 for negative values
    if n < 0:
        return 0

    # There is only 1 way to reach a score of 0
    if n == 0:
        return 1

    # Recursively calculate the number of ways using the
    # recurrence relation we saw earlier
    return scoring_options(n - 1) + scoring_options(n - 2) + scoring_options(n - 4)
# Optimized
# -- Top down
def scoring_options_rec(n, memo):
    # We can not get a negative score, we return 0 for negative values
    if n < 0:
        return 0

    # Check if a solution already exists in the array
    if memo[n] != -1:
        return memo[n]

    # Else, we recursively calculate the solution for the
    # given value and store it in our solution array

    memo[n] = scoring_options_rec(n - 1, memo) + scoring_options_rec(n - 2, memo) + scoring_options_rec(n - 4, memo)

    return memo[n]

# Scoring options are 1, 2, and 4
def scoring_options(n):

    # Initializing our solution array
    memo = [-1] * max(5, n + 1)

    # Set up the base case, 1 way to score 0
    memo[0] = 1

    # Pass our array to the helper function
    return scoring_options_rec(n, memo)

# -- Bottom up
# Scoring options are 1, 2, and 4
def scoring_options(n):
    # Initializing our solution array
    dp = [0] * (n + 1)

    # Each array index holds the number of ways to
    # reach a score equal to the index
    dp[0] = 1

    # Variables to store scores
    s1 = s2 = s4 = 0

    # Iteratively calculate the number of ways to reach a
    # score and store it into the solutions' array
    for r in range(1, n+1):
        # Return 0 if index is less than 0, otherwise
        # set to array value
        s1 = 0 if r - 1 < 0 else dp[r - 1]
        s2 = 0 if r - 2 < 0 else dp[r - 2]
        s4 = 0 if r - 4 < 0 else dp[r - 4]

        # Using our recurrence relation to calculate new answers
        dp[r] = s1 + s2 + s4
    return dp[n]

## Unique Paths to Goal
'''
Given a robot located at the top-left corner of an m×n matrix, determine the number of unique paths the robot can take from start to finish while avoiding all obstacles on the matrix.
The robot can only move either down or right at any time. The robot tries to reach the bottom-right corner of the matrix.
An obstacle is marked as 1, and an unoccupied space is marked as 0 in the matrix.
Naive approach: Every time, we need to check if we have reached the bottom-right corner of the matrix or if the iterating index exceeds the size of the matrix. If this condition is satisfied, we just stop the iteration here. If an obstacle is present in the array, the number of unique paths will be 0 up to that obstacle. After that, we will traverse to the next cell in the row and the column finding all the possible unique paths.
O(2^mn) - O(mn)
---
Optimized solution
Top-down solution: O(mn) - O(mn)
Bottom-up solution: Here we again create a 2-D array of the same size as the matrix given. In the bottom-up approach, we start moving through the rows and filling in the values. If an obstacle (1) is found at any place, we will set the value of that cell to 0 because we cannot reach that place, so unique paths will be 0. Starting with the base condition, we will set the values of the first row and column to 1 if no obstacles are found. Now we will start traversing row-wise, if there is no obstacle, then the answer will be the sum of the values of the top and left cells, and if there is an obstacle, we will just insert the value 0 to that cell irrespective of the values of other cells.
O(mn) - O(1)
'''
# Naive
def find_unique_path(matrix):
  # the length of 2d matrix will be equal to the number of rows
  rows = len(matrix)  

  # The number of elements in 1st row are equal to the number of columns in 2d matrix
  column = len(matrix[0]) 

  return find_unique_path_recursive(0, 0, rows, column, matrix)

# Helper function to check the boundaries and base case
def find_unique_path_recursive(i, j, row, col, matrix):

  # check the boundary constraints
  if (i == row or j == col):
    return 0

  # check if obstacle is present or not
  if (matrix[i][j] == 1):
    return 0

  # check the base case when the last cell is reached
  if (i == row-1 and j == col-1):
    return 1
  
  # using the recursive approach when moving to next row or next column 
  return find_unique_path_recursive(i+1, j, row, col, matrix) + find_unique_path_recursive(i, j+1, row, col, matrix)

# Optimized
# -- Top down
def find_unique_path(matrix):
    # the length of 2d matrix will be equal to the number of rows
    rows = len(matrix)  

    # The number of elements in 1st row are equal to the number of columns in 2d matrix
    column = len(matrix[0]) 

    pathArray = [[-1 for index1 in range(column)]for index2 in range(rows)]

    return find_unique_path_memoization(0, 0, rows, column, matrix, pathArray)

# Helper function to check the boundaries and base case
def find_unique_path_memoization(i, j, row, cols, matrix, pathArray):

  # check the boundary constraints
  if (i == row or j == cols):
    return 0
  
  # check if obstacle is present or not 
  if (matrix[i][j] == 1):
    return 0
  
  # check the base case when the last cell is reached
  if (i == row-1 and j == cols-1):
    return 1
    
  if (pathArray[i][j] != -1):
    return pathArray[i][j]
 
  pathArray[i][j] = find_unique_path_memoization(i+1, j, row, cols, matrix, pathArray) + find_unique_path_memoization(i, j+1 , row, cols, matrix, pathArray)
  return pathArray[i][j]

# -- Bottom up
def find_unique_path(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # If the starting cell has an obstacle, then return 0 
    # as there would be no paths to the destination.
    if matrix[0][0] == 1:
        return 0

    # Number of ways of reaching the starting cell = 1.
    matrix[0][0] = 1

    # Fill the values for the first column
    for i in range(1, rows):
        matrix[i][0] = int(matrix[i][0] == 0 and matrix[i-1][0] == 1)

    # Fill the values for the first row        
    for j in range(1, cols):
        matrix[0][j] = int(matrix[0][j] == 0 and matrix[0][j-1] == 1)

    # Start from matrix[1][1], we fill up the values.
	# The number of ways of reaching matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1]
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][j] = matrix[i-1][j] + matrix[i][j-1]
            else:
                matrix[i][j] = 0

    # Return value stored in rightmost bottommost cell. 
	# That is the destination.
    return matrix[rows - 1][cols - 1]

## Nth Tribonacci Number
'''
Tribonacci numbers are a sequence of numbers where each number is the sum of the three preceding numbers. Your task is to find the nth Tribonacci number.
Naive approach: We can solve this problem with the following steps:
If 0th Tribonacci number is required, we will return 0, as it is one of the base cases.
If 1st or 2nd Tribonacci number is required, we will return 1, as these are also the base cases.
Otherwise, we’ll call the function recursively three times to compute the next number’s sum, as the Tribonacci number is the sum of the previous three numbers. We’ll repeat this process until we reach the base case.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: In the recursive approach, we are calling the tribonacci function on the previous three numbers. So for a value of n, there might be multiple recursive calls for evaluating the same nth Tribonacci number. Therefore, we need an array with size n+1 to store the result of the nth Tribonacci number at the index n when we encounter it for the first time. At any later time, if we encounter the same number n, we can fetch the stored result from the array in a constant time instead of recalculating the values.
Bottom-up solution: we follow the algorithm as:
- If n is less than 3, the result is determined by the base cases.
- Otherwise, declare an array of size n+1 and initialize the first three Tribonacci numbers.
- Then we compute the 3rd Tribonacci number based on the sequence formula, then the 4th, and so on, iteratively till we reach the nth Tribonacci number.
'''
# Naive
def tribonacci(n):
    # base cases
    if n == 0:
        return 0
    elif n == 1 or n == 2: 
        return 1
    # recursive case
    return tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3)

# Optimized
# -- Top down
def tribonacci(n):
    cache = [0] * (n + 1)
    return tribonacciHelper(n, cache)

def tribonacciHelper(n, cache):
    # Base cases
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    # Using memoization table to get the already evaluated result
    if cache[n] != 0:
        return cache[n]
    
    cache[n] = tribonacciHelper(n - 1, cache) + \
                tribonacciHelper(n - 2, cache) + \
                tribonacciHelper(n - 3, cache)
    return cache[n]

# -- Bottom up
def tribonacci(n):
    # Base cases
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    
    # Creating an array
    dp = [0] * (n + 1)
    # First three tribonacci numbers
    dp[1], dp[2] = 1, 1
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
    
    return dp[n]

## The Catalan Numbers
'''
Given a number n, find the nth Catalan number.
Naive approach: O(3^n) - O(n)
---
Optimized Solution
Top-down solution: O(n^2) - O(n)
Bottom-up solution: O(n^2) - O(n)
'''
# Naive
def catalan(n):
  # base case; C(0) = 1
  if n == 0:      
    return 1
  sum = 0
  # iterate from 1...n to evaluate: C(0)*C(n-1) + C(1)*C(n-2) ... + C(n-1)*C(0)
  for i in range(n):  
    # C(i)*C(n-1-i)
    sum += (catalan(i) * catalan(n-1-i))  
  return sum

# Optimized
# -- Top down
def catalan_memo(n, memo):
  # base case; C(0) = 1
  if n == 0:          
    return 1
  # if n already evaluated, return from memo
  elif n in memo:       
    return memo[n]
  sum = 0
  # iterate from 1...n to evaluate: C(0)*C(n-1) + C(1)*C(n-2) ... + C(n-1)*C(0)
  for i in range(n):  
     # C(i)*C(n-1-i)
    sum += (catalan_memo(i, memo) * catalan_memo(n-1-i, memo)) 
  # store result in memo
  memo[n] = sum         
  return memo[n]


def catalan(n):
  memo = {}
  return catalan_memo(n, memo)

# -- Bottom up
def catalan(n):
  # tabulating 
  dp = [None] * (n+1)  
  # handling the base case
  dp[0] = 1            
  # iterating to fill up the tabulation table
  for i in range(1,n+1):  
    # initializing the i-th value to 0
    dp[i] = 0          
    # iterate from 0 to i; according to formula of catalan i.e. 
    # C0*Ci + C1*Ci-1 + ... Ci*C0
    for j in range(i):
      # C(j) * C(i-j-1)    
      dp[i] += (dp[j] * dp[i-j-1]) 
  return dp[n]  

## House Thief Problem
'''
You are a professional robber, and after weeks of planning, you plan to rob some houses along a street. Each of these houses has a lot of money and valuables. Let’s say that you cannot rob houses adjacent to each other since they have security and burglar alarms installed.
Following the above-mentioned constraint and given an integer array, money, that represents the amount of money in each house, return the maximum amount of money you can steal tonight without alerting the police.
Naive approach: O(2^n) - O(n)
---
Optimized solution
Top-down approach: O(n) - O(n) 
Bottom-up approach: initialize an array of size one greater than the length of money with zeros. We copy the money from the first house money[0] to the memo[1] as it will serve as the maximum theft value in the first iteration. Next, we will iterate over the money array from index 1 and do the following in each iteration i:
- Compute the maximum between memo[i] and the combined value money[i] + memo[i - 1].
- Store the maximum on memo[i+1].
O(n) - O(n) 
'''
# Naive
def house_thief(money):
  #Call helper function
  return house_thief_rec(money, 0)

#Helper function
def house_thief_rec(money, ind):
  #Stopping criteria
  stop = len(money)
  #Base case
  if ind >= stop:
    return 0
  new_ind = ind+1
  #returns the maximum of the following two values:
  #1. Leaving the current house and starting from the next house
  #2. The maximum robbery if we rob the current house along with the houses starting from the next to
  # the next house, as we cannot rob adjacent houses
  return max(house_thief_rec(money,new_ind), money[new_ind-1] + house_thief_rec(money, new_ind+1))

# Optimized
# -- Top down
def house_thief(money):
  #Stopping criteria
  stop = len(money)
  #Store values in memo
  memo = [0] * stop
  return house_thief_(memo, money, 0)

#Helper function
def house_thief_(memo, money, ind):
  stop = len(money)
  #Base case
  if ind >= stop:
    return 0
    
  if memo[ind] == 0:
    new_ind = ind+1
   #stores the maximum of the following two values at memo[ind]:
   #1. Leaving the current house and starting from the next house
   #2. The maximum robbery if we rob the current house along with the houses starting from the next to
   # the next house
    memo[ind] = max(house_thief_(memo, money, new_ind), money[new_ind-1] + house_thief_(memo, money, new_ind+1))
  return memo[ind]

# -- Bottom up
def house_thief(money):
  #Stopping criteria
  stop = len(money)
  #Base case
  if stop == 0:
    return 0

  memo = [0] * (stop+1)
  memo[1] = money[0]  
  for i in range(1, stop):
    #stores the maximum of the following two values at memo[i+1]:
    #1. The combined sum of money in the current house alongwith 
    #the maximum theft possible from the previous of the previous house.
    #2. The maximum theft possible till the previous house, we should refer to memo to get this value.
    memo[i + 1] = max(money[i] + memo[i - 1], memo[i])
  return memo[stop]

## Minimum Jumps to Reach the End
'''
Given an array nums of positive numbers, start from the first index and reach the last index with the minimum number of jumps, where a number at an index represents the maximum jump from that index.
Naive approach: Suppose the value at the 0th index is j. We will try all the jumping combinations (0 to j). After jumping, we will recursively call the function and try all the jumping options again from that particular index on which we have jumped. jumps is initialized with math.inf, which will be updated with the minimum number of jumps required till that index. In the end, we will have the minimum number of jumps to reach the end stored in jumps.
O(2^n) - O(n)
---
Optimized Solution
Top-down solution: O(n^2) - O(n)
Bottom-up solution: O(n^2) - O(n)
'''
# Naive
def find_min_jumps(nums):
  return find_min_jumps_recursive(nums, 0)

def find_min_jumps_recursive(nums, index):
  n = len(nums)
  # If we have reached the last index
  if (index >= n - 1):
      return 0

  # Initializing jumps with maximum value. It will store 
  # the minimum jumps required to reach the current index
  jumps = math.inf

  # Checking all the possible jumps
  for i in range(index + 1, index + nums[index] + 1):
    # Selecting the minimum jump
    jumps = min(jumps, find_min_jumps_recursive(nums, i) + 1)
  
  return jumps

# Optimized
# -- Top down
def find_min_jumps(nums):
  # Initializing the lookup table of nums length
  lookup = [math.inf] * (len(nums))
  return find_min_jumps_memo(lookup, nums, 0)

def find_min_jumps_memo(lookup, nums, index):
  n = len(nums)
  # If we have reached the last index
  if (index >= n - 1):
      return 0

  # If we have already solved this problem, return the result
  if lookup[index] != math.inf:
    return lookup[index]

  # Initializing jumps with maximum value. It will store 
  # the minimum jumps required to reach the current index
  jumps = math.inf

  # Checking all the possible jumps
  for i in range(index + 1, index + nums[index] + 1):
    # Selecting the minimum jump
    jumps = min(jumps, find_min_jumps_memo(lookup, nums, i) + 1)
  
  # Storing the value in lookup table
  lookup[index] = jumps
  return lookup[index]

# -- Bottom up
def find_min_jumps(nums):
  n = len(nums)
  # Initializing the lookup table of nums length
  lookup = [math.inf] * (n)
  # Setting the 0th index to 0
  lookup[0] = 0
  
  # Outer loop traversing the whole array
  for i in range(1, n):
    # Inner loop traversing from 0 to the ith index
    for j in range(i):
      # If the value is not stored in the table and index i is
      # less than equal to the value at jth index + j index
      if ((i <= (nums[j] + j)) and (lookup[j] != math.inf)):
        lookup[i] = min(lookup[i], lookup[j] + 1)
        break
      
  return lookup[n - 1]

## Minimum Jumps With Fee
'''
You are given n steps of stairs and a list fee because each step has a fee associated with it. Your task is to calculate the minimum fee required to reach the top of the stairs (beyond the top step), assuming you start with the first step. At every step, you can take 1 step, 2 steps, or 3 steps.
Naive approach: We would then return the combination with the minimum fee:
- If you hop 1 step, you have n−1 remaining steps.
- If you hop 2 steps, then you have n−2 remaining steps.
- If you hop 3 steps, then you have n−3 remaining steps.
If you hop 1 step, you can hop 1, 2, or 3 steps again until n equals 0.
O(3^n) - O(n)
---
Optimized Solution
Top-down solution: O(n) - O(n)
Bottom-up solution: O(n) - O(n)
'''
# Naive
def min_fee(fee, n):
  #Check if the number of steps are 0
  if n < 1:
    return 0

  #calculate fee for each choice
  one_step = fee[n-1] + min_fee(fee, n - 1)
  two_step = fee[n-2] + min_fee(fee, n - 2)
  three_step = fee[n-3] + min_fee(fee, n -3 )

  #return the minimum fee from the three choices calculated above
  return min(one_step, two_step, three_step)

# Optimized
# -- Top down
def min_fee_rec(fee, n, lookup_array):
  # If the number of steps get to 0
  if n < 1:
    return 0

  # Check if the fee has already been calculated and stored in the lookup array
  elif lookup_array[n] > -1:
    return lookup_array[n]
  
  # Find the fee for each step and then storing the minimum in the array
  else:
    one_step = fee[n - 1] + min_fee_rec(fee, n - 1, lookup_array)
    two_step = fee[n - 2] + min_fee_rec(fee, n - 2, lookup_array)
    three_step = fee[n - 3] + min_fee_rec(fee, n - 3, lookup_array)
    lookup_array[n] = min(one_step, two_step, three_step)
    
  return lookup_array[n]

def min_fee(fee, n):
  lookup_array = [-1 for x in range(n + 1)]
  return min_fee_rec(fee, n, lookup_array)

# -- Bottom up
def min_fee(fee, n):

  #Create a lookup array
  lookup_array = [0 for x in range(n + 1)]

  # Add the initial values to the array
  lookup_array[0] = 0
  lookup_array[1] = fee[0]
  lookup_array[2] = fee[0] 

  for i in range(3, n + 1):
    # Fill up the table by finding the minimum of the previous three values
    one_step = fee[i - 1] + lookup_array[i - 1]
    two_step = fee[i - 2] + lookup_array[i - 2]
    three_step = fee[i-3] + lookup_array[i - 3]
    lookup_array[i] = min(one_step, two_step, three_step)
    
  return lookup_array[n]

## Matrix Chain Multiplication
'''
You are given a chain of matrices to be multiplied. You have to find the least number of primitive multiplications needed to evaluate the result.
Naive approach: We can easily imagine dims to be a list of matrices. Then each recursive call tries to find the optimal placement of two parentheses between these matrices. So, let’s say we have the following sequence of matrices: A1A2A3A4. Each recursive call tries to place two parentheses, so we have the following possibilities:
- (A1)A2A3A4
- (A1A2)A3A4
- (A1A2A3)A4
Therefore, we make recursive calls for all these possibilities and finally choose the best one from among them.
O(n2^n) - O(n)
---
Optimized solution
Top-down solution:  The important bit is our choice of key for memoization, which we have created by using a tuple of i and j. Since i and j can uniquely identify a subarray from dims, a tuple of these two variables fits perfectly for the key. If we had used an indexing approach to get the subarray, we wouldn’t be able to memoize our results. This is because lists cannot be used as a key in dictionaries.
O(n^3) - O(n^2)
Bottom-up solution: the dp table, it is a 2-D list of dimensions n×n (where � n is the length of dims), where dp[i][j], for any i, j <n , denotes the minimum number of multiplications required to multiply a chain of matrices formed between i and j. Now, we need to find a sequence in which we fill the dp table so that no value is needed before it is evaluated. We know from our previous solution that we should start from the base case of a single matrix’s multiplication. This we have covered in our initialization of dp by setting everything to 0 0 . Next, we need to handle the cases for multiplications of the chains of size 2 2 , since these will be used by all the bigger problems. After 2 2 , we will need to fill for chains of size 3 3 , and so on. The nested for loops are simply calculating the optimal answer in the same way as the previous solutions, i.e., by finding the minimum cumulative value from all the subproblems.
O(n^3) - O(n^2)
'''
# Naive
def min_multiplications(dims):
    # Base case
    if len(dims) <= 2:
        return 0
    minimum = math.inf  # Init with maximum integer value
    for i in range(1,len(dims)-1): # Recursive calls for all the possible multiplication sequences
        minimum = min(minimum, 
                    min_multiplications(dims[0:i+1]) +  # solve the subproblem up to the ith matrix
                    min_multiplications(dims[i:]) +     # solve the subproblem from the i+1th matrix to the last
                    dims[0] * dims[i] * dims[-1])       # calculate the number of multiplications for the 
                                                        # current pair of matrices with dimensions dims[0]xdims[i] and dims[i]xdims[-1]
    return minimum

# Optimized
# -- Top down
def min_multiplications_recursive(dims, i, j):
    # Base case
    if j-i <= 2:
        return 0
    minimum = math.inf # Init with maximum integer value
    for k in range(i+1, j-1): # Recursive calls for all the possible multiplication sequences
        minimum = min(minimum, min_multiplications_recursive(dims, i, k+1) + # solve the subproblem from ith up to the kth matrix
        min_multiplications_recursive(dims, k, j) +                          # solve the subproblem from the k+1th matrix to the jth matrix
        dims[i] * dims[k] * dims[j-1])                                       # calculate the number of multiplications for the 
                                                                             # current pair of matrices with dimensions dims[i]xdims[k] and dims[k]xdims[j-1]
    return minimum

def min_multiplications(dims):
    return min_multiplications_recursive(dims, 0, len(dims))

# -- Bottom up
def min_multiplications_recursive(dims, i, j, memo):
    # Base case
    if j-i <= 2:
        return 0
    if (i,j) in memo:
        return memo[(i,j)]
    minimum = math.inf # Init with maximum integer value
    for k in range(i+1, j-1): # Recursive calls for all the possible multiplication sequences
        minimum = min(minimum, min_multiplications_recursive(dims, i, k+1, memo) + # solve the subproblem from ith up to the kth matrix
        min_multiplications_recursive(dims, k, j, memo) +                          # solve the subproblem from the k+1th matrix to the jth matrix
        dims[i] * dims[k] * dims[j-1])                                             # calculate the number of multiplications for the 
                                                                                   # current pair of matrices with dimensions dims[0]xdims[i] and dims[i]xdims[-1]
    memo[(i,j)] = minimum # Storing the minimum value to the memo
    return minimum

def min_multiplications(dims):
    memo = {}
    return min_multiplications_recursive(dims, 0, len(dims), memo)

### *** Longest Common Substring
## Longest Common Substring
'''
Given two strings s1 and s2, you have to find the length of the Longest Common Substring (LCS) in both these strings.
Let’s say we have two strings, “helloworld” and “yelloword”, there are multiple common substrings, such as “llo”, “ello”, “ellowor”, “low”, and “d”. The longest common substring is “ellowor”, with length 7.
Naive approach: We are comparing each character one by one with the other string. There can be three possibilities for the ith character of s1 and the jth character of s2:
- If both characters match, these could be part of a common substring, meaning we should count this character towards the length of the longest common substring.
- ithcharacter of s1 might match with (j+1)th character of s2.
- jth character of s2 might match with (i+1)th character of s1.
Therefore, we take the maximum among all three of these possibilities.
O(3^(m+n)) - O(m + n)
---
Optimized solution
Top-down solution: We have three parameters, i, j, and count, that uniquely define every subproblem. Therefore, we can address each subproblem as (i, j, count) using a hashmap. The first time we encounter the subproblem (i, j, count), we compute its result and store it in memo[(i, j, count)]. 
O(mn^2) - O(mn^2)
Bottom-up solution: We start off by constructing a table of size m×n for tabulation, where n is the size of s1 and m is the size of s2. We call this table dp and initialize it to zeros. Now we start filling the table starting from position 1, 1. Each entry in this table tells us the count of the last characters matched between both strings up to ith and jth positions in s1 and s2 respectively. So for example, if dp[3][4] contains 2, this means the last two characters of s1 and s2 up to positions 3 and 4 match, i.e., s1[2] = s2[3] and s1[1] = s2[2]. By the end of the execution of this algorithm, we will have the length of the longest common substring in the variable max_length, since, at each step, we update max_length in case it's less than the current entry in the dp table.
O(mn) - O(mn)
'''
# Naive
def lcs_length_rec(s1, s2, i, j, count):
  # base case of when either string has been exhausted
  if i >= len(s1) or j >= len(s2):  
    return count
  # if i and j characters match, increment the count and compare the rest of the strings
  if s1[i] == s2[j]:     
    count = lcs_length_rec(s1, s2, i+1, j+1, count+1)
  # compare s1[i:] with s2, s1 with s2[j:], and take max of current count and these two results
  return max(count, lcs_length_rec(s1, s2, i+1, j, 0), lcs_length_rec(s1, s2, i, j+1, 0))
  

def lcs_length(s1, s2):
  return lcs_length_rec(s1, s2, 0, 0, 0)

# Optimized
# -- Top down
def lcs_length_rec(s1, s2, i, j, count, memo):
  # base case of when either string has been exhausted
  if i >= len(s1) or j >= len(s2):  
    return count
  # check if result available in memo
  if (i,j,count) in memo:       
    return memo[(i,j,count)]
  c = count
   # if i and j characters match, increment the count and compare the rest of the strings
  if s1[i] == s2[j]:     
    c = lcs_length_rec(s1, s2, i+1, j+1, count+1, memo)
  # compare s1[i:] with s2, s1 with s2[j:], and take max of current count and these two results
  # memoize the result
  memo[(i,j,count)] = max(c, lcs_length_rec(s1, s2, i+1, j, 0, memo), lcs_length_rec(s1, s2, i, j+1, 0, memo))
  return memo[(i,j,count)]

def lcs_length(s1, s2):
  memo = {}
  return lcs_length_rec(s1, s2, 0, 0, 0, memo)

# -- Bottom up
def lcs_length(s1, s2):
  n = len(s1)   # length of s1
  m = len(s2)   # length of s2

  dp = [[0 for j in range(m+1)] for i in range(n+1)]  # table for tabulation of size m x n
  max_length = 0   # to keep track of longest substring seen 

  for i in range(1, n+1):           # iterating to fill table
    for j in range(1, m+1):
      if s1[i-1] == s2[j-1]:    # if characters at this position match, 
        dp[i][j] = dp[i-1][j-1] + 1 # add 1 to the previous diagonal and store it in this diagonal
        max_length = max(max_length, dp[i][j])  # if this substring is longer, update max_length
      else:
        dp[i][j] = 0 # if character don't match, common substring size is 0
  return max_length

## Longest Common Subsequence
'''
Suppose you are given two strings. You need to find the length of the longest common subsequence between these two strings.
A subsequence is a string formed by removing some characters from the original string, while maintaining the relative position of the remaining characters. For example, “abd” is a subsequence of “abcd”, where the removed character is “c”.
If there is no common subsequence, then return 0.
Naive approach: A naive approach is to compare the characters of both strings based on the following rules:
- If the current characters of both strings match, we move one position ahead in both strings.
- If the current characters of both strings do not match, we recursively calculate the maximum length of moving one character forward in any one of the two strings i.e., we check if moving a character forward in either the first string or the second will give us a longer subsequence.
- If we reach the end of either of the two strings, we return 0.
O(2^(m+n)) - O(m + n)
---
Optimized Solution
Top-down solution: In the recursive approach, the following two variables kept changing:
- The index i, used to keep track of the current character in str1.
- The index j, used to keep track of the current character in str2.
We will use a 2-D table, dp, with n rows and m columns to store the result at any given state.  
O(mn) - O(mn)
Bottom-up solution: We first make a 2-D array of size [(m+1)×(n+1)], where n is the length of str1 and m is the length of str2. This array is initialized to 0. We need the first row and column to be 0 for the base case. Any entry in this array given by dp[i][j] is the length of the longest common subsequence between str1 up to ith position and str2 up to the jth position.
As we saw in the recursive algorithm, when the characters matched, we could simply move one position ahead in both strings and add one to the result. This is exactly what we do here as well. We add 1 to the previous diagonal result. In case we have a mismatch, we need to take the maximum of two subproblems:
dp[i][j] = max(dp[i-1][j], dp[i][j-1]) 
We could either move one position ahead in str1 (the subproblem dp[i-1][j]), or we could move one step ahead in str2 (the subproblem dp[i][j-1]). In the end, we have the optimal answer for str1 and str2 in the last position, i.e., dp[n][m].
O(mn) - O(mn)
'''
# Naive
# Helper function with updated signature: i is current index in str1, j is current index in str2
def longest_common_subsequence_helper(str1, str2, i, j): 
    # base case
    if i == len(str1) or j == len(str2): 
        return 0

    # if current characters match, increment 1
    elif str1[i] == str2[j]:  
        return 1 + longest_common_subsequence_helper(str1, str2, i+1, j+1)
    
    # else take max of either of two possibilities
    return max(longest_common_subsequence_helper(str1, str2, i+1, j), longest_common_subsequence_helper(str1, str2, i, j+1))

def longest_common_subsequence(str1, str2):
    return longest_common_subsequence_helper(str1, str2, 0, 0)
# Optimized
# -- Top down
# Helper function with updated signature: i is current index in str1, j is current index in str2
def longest_common_subsequence_helper(str1, str2, i, j, dp): 
    # base case
    if i == len(str1) or j == len(str2): 
        return 0

    elif dp[i][j] != -1:
        return dp[i][j]

    # if current characters match, increment 1
    elif str1[i] == str2[j]:  
        dp[i][j] = 1 + longest_common_subsequence_helper(str1, str2, i+1, j+1, dp)
        return dp[i][j]
    
    # else take max of either of two possibilities
    dp[i][j] = max(longest_common_subsequence_helper(str1, str2, i+1, j, dp), 
                   longest_common_subsequence_helper(str1, str2, i, j+1, dp))
    return dp[i][j]

def longest_common_subsequence(str1, str2):
    n = len(str1)
    m = len(str2)
    dp = [[-1 for x in range(m)] for y in range(n)]
    return longest_common_subsequence_helper(str1, str2, 0, 0, dp)
# -- Bottom up
def longest_common_subsequence(str1, str2):
    n = len(str1)   # length of str1
    m = len(str2)   # length of str2

    rows = n + 1
    cols = m + 1

    # Initializing the 2-D table, filling the first row and column with all 0s
    dp = [[0 if (i == 0 or j == 0) else -1 for i in range(cols)] for j in range(rows)]

    # Iterating to fill the table
    for i in range(1, rows):           
        # calculate new row (based on previous row i.e. dp)
        for j in range(1, cols):
            # if characters at this position match, 
            if str1[i-1] == str2[j-1]:    
                # add 1 to the previous diagonal and store it in this diagonal
                dp[i][j] = dp[i-1][j-1] + 1 
            else:
                # If the characters don't match, fill this entry with the max of the
                # left and top elements
                dp[i][j] = max(dp[i][j-1], dp[i-1][j]) 
    return dp[n][m]

## Shortest Common Supersequence
'''
Given two strings, str1 and str2, find the length of the shortest string that has both the input strings as subsequences.
Naive approach: The two changing parameters in our recursive function are the two indices, i1 and i2 which are incremented based on the following two conditions:
- If the characters at i1 and i2 are a match, skip one character from both the sequences and make a recursive call for the remaining lengths.
- If the characters do not match, call the function recursively twice by skipping one character from each string. Return the minimum result of these two calls.
O(2^(n+m)) - O(n+m)
---
Optimized solution
Top-down solution: O(mn) - O(mn)
Bottom-up solution: We create a lookup table of size (m+1)×(n+1). We fill the 0th row with values from 0 to m and similarly the 0th column with values from 0 to n. The constants in the 0th row tell the length of the shortest common supersequence if str2 is empty. Likewise, the constants in the 0th column tell the length of the shortest common supersequence if str1 is empty. Using these pre-filled values in the lookup table we iterate through both strings in a nested for loop, starting i1 and i2 from 1, and check if any of the two conditions are TRUE:
- If the character str1[i1-1] matched str2[i2-1], the length of the shortest common supersequence would be one plus the length of the shortest common supersequence till i1 - 1 and i2 - 1 indices in the two strings.
- If the character str1[i1-1] does not match str2[i2-1], we consider two shortest common supersequences - one without str1[i1-1] and one without str2[i2-1]. Our required shortest common supersequence length is the shortest of these two super sequences plus one.
O(mn) - O((m+1)*(n+1))
'''
# Naive
def shortest_common_supersequence_recursive(str1, str2, i1, i2):
    # if any of the pointers has iterated over all the characters of the string,
    # return the remaining length of the other string 
    if i1 == len(str1):
        return len(str2) - i2
    if i2 == len(str2):
        return len(str1) - i1

    # if both the characters pointed by i1 and i2 are same, increment both pointers
    if str1[i1] == str2[i2]:
        return 1 + shortest_common_supersequence_recursive(str1, str2, i1 + 1, i2 + 1)

    # recursively call the function twice by skipping one character from each string
    length1 = 1 + shortest_common_supersequence_recursive(str1, str2, i1, i2 + 1)
    length2 = 1 + shortest_common_supersequence_recursive(str1, str2, i1 + 1, i2)

    # return the minimum of the two lengths
    return min(length1, length2)

def shortest_common_supersequence(str1, str2):
    return shortest_common_supersequence_recursive(str1, str2, 0, 0)

# Optimized
# -- Top down
def shortest_common_supersequence_recursive(lookup_table, str1, str2, i1, i2):
    # if any of the pointers has iterated over all the characters of the string,
    # return the remaining length of the other string 
    if i1 == len(str1):
        return len(str2) - i2
    if i2 == len(str2):
        return len(str1) - i1

    # check if the value for the current pointers is already present in the lookup table
    if lookup_table[i1][i2] == 0:
        # if both the characters pointed by i1 and i2 are same, increment both pointers
        if str1[i1] == str2[i2]:
            lookup_table[i1][i2] = 1 + shortest_common_supersequence_recursive(lookup_table, str1, str2, i1 + 1, i2 + 1)
        else:
            # recursively call the function twice by skipping one character from each string
            length1 = 1 + shortest_common_supersequence_recursive(lookup_table, str1, str2, i1, i2 + 1)
            length2 = 1 + shortest_common_supersequence_recursive(lookup_table, str1, str2, i1 + 1, i2)
            lookup_table[i1][i2] = min(length1, length2)

    # return the value stored in the lookup table
    return lookup_table[i1][i2]

def shortest_common_supersequence(str1, str2):
    # lookup table to store the values of recursive calls to prevent redundancy
    lookup_table = [[0 for x in range(len(str2))] for x in range(len(str1))]
    return shortest_common_supersequence_recursive(lookup_table, str1, str2, 0, 0)

# -- Bottom up
def shortest_common_supersequence(str1, str2):
    # lookup table of size (m+1)*(n+1)
    lookup_table = [[0 for x in range(len(str2) + 1)] for x in range(len(str1) + 1)]

    # filling the additional row and column with constants
    for i in range(len(str1) + 1):
        lookup_table[i][0] = i

    for j in range(len(str2) + 1):
        lookup_table[0][j] = j

    # iterate through both the strings
    for i1 in range(1, len(str1) + 1):
        for i2 in range(1, len(str2) + 1):
            # if the characters match or not
            if str1[i1 - 1] == str2[i2 - 1]:
                lookup_table[i1][i2] = 1 + lookup_table[i1 - 1][i2 - 1]
            else:
                lookup_table[i1][i2] = 1 + min(lookup_table[i1 - 1][i2], lookup_table[i1][i2 - 1])

    # return the last value of the lookup table
    return lookup_table[len(str1)][len(str2)]

## Minimum Number of Deletions and Insertions
'''
Given two strings, str1 and str2, find the minimum number of deletions and insertions required to transform str1 into str2.
Naive approach: Starting from the first character of both strings, there are three possibilities for every pair of characters being considered:
The first characters of both the strings match, in which case, we increment the count of the matching characters by 1. We move one position ahead in both strings, and the function is called recursively on the remaining strings.
The first characters of both the strings do not match, in which case, the following possibilities are checked on both strings:
- A recursive call by moving one position ahead in the str1
- A recursive call by moving one position ahead in the str2
- Return the maximum of the matching characters found by both the calls
If we reach the end of either of the two strings, we return 0.
After finding the n (maximum number of the matching characters subsequence), we can find the number of deletions required on str1 to transform it into str2 by subtracting n from the length of str1. The number of insertions required on str1 to transform it into str2 can be found by subtracting n from the length of str2.
O(2^n) - O(n)
---
Optimized Solution
Top-down solution: for each character, the value of the maximum number of matching characters is stored in the lookup_table.
If the start characters match, the value returned is stored at lookup_table[i][j].
If the start characters don’t match, in which case, the following possibilities are checked on both strings:
- A recursive call by moving one position ahead in the str1
- A recursive call by moving one position ahead in the str2
- Store the maximum of the matching characters found by both the calls in the lookup_table[i][j] and return it
O(n^2) - O(n)
Bottom-up solution: we build the solution bottom-up by dividing our problem into subproblems.
We first make a 2-D table lookup_table[m+1][n+1] where m is the length of str1 and n is the length of str2. This table is initialized with 0s. We need the first row and column to be 0 for the base case. Any entry in this table given by lookup_table[i][j] is the maximum number of matching characters subsequence between str1 up till ith position and str2 up to the jth position.
If the characters match, we store the returned values from lookup_table[i-1][j-1] in lookup_table[i][j].
For characters that don’t match, we need to take a maximum of two subproblems:
- Move one position ahead in str1 (the subproblem lookup_table[i-1][j])
- Move one position ahead in str2 (the subproblem lookup_table[i][j-1])
In the end, we have the optimal answer for the maximum number of matching characters subsequence for str1 and str2 in the last position, i.e., lookup_table[m][n].
With this, we can easily calculate the minimum number of deletions and insertions required to transform str1 into str2
O(n^2) - O(mn)
'''
# Naive
# function to find the maximum number of matching characters subsequence
def find_max_matching_subseq(str1, str2, i, j): 
    # base case
    if i == len(str1) or j == len(str2): 
        return 0

    # if current characters match, increment by 1
    elif str1[i] == str2[j]:  
        return 1 + find_max_matching_subseq(str1, str2, i+1, j+1)
    
    # else return max of either of two possibilities
    return max(find_max_matching_subseq(str1, str2, i+1, j), find_max_matching_subseq(str1, str2, i, j+1))


def min_del_ins(str1, str2):
  n = find_max_matching_subseq(str1, str2, 0, 0)
  # calculating number of deletions required from str1 to transform it into str2
  deletions = len(str1) - n
  # calculating number of insertions required in str1 to transform it into str2
  insertions = len(str2) - n

  return deletions,insertions
 
# Optimized
# -- Top down
# function to find the maximum number of matching characters subsequence
def find_max_matching_subseq(str1, str2, i, j, lookup_table): 

    # base case
    if i == len(str1) or j == len(str2): 
        return 0

    # if the subproblem has been computed before, return the value stored in lookup_table
    elif lookup_table[i][j] != -1: 
        return lookup_table[i][j]

    # if current characters match, increment by 1
    elif str1[i] == str2[j]:  
        lookup_table[i][j] = 1 + find_max_matching_subseq(str1, str2, i+1, j+1, lookup_table)
        return lookup_table[i][j]
        
    # else take max of either of two possibilities
    lookup_table[i][j] = max(find_max_matching_subseq(str1, str2, i+1, j, lookup_table), find_max_matching_subseq(str1, str2, i, j+1, lookup_table))
    return lookup_table[i][j]

def min_del_ins(str1, str2):
  # Declare a lookup_table array which stores the answer to recursive calls  
  lookup_table = [[-1 for i in range(len(str2))] for i in range(len(str1))] 
   
  n = find_max_matching_subseq(str1, str2, 0, 0, lookup_table)
  # calculating number of deletions required from str1 to transform it into str2
  deletions = len(str1) - n
  # calculating number of insertions required in str1 to transform it into str2
  insertions = len(str2) - n
  
  return deletions,insertions

# -- Bottom up
# function to find the maximum number of matching characters subsequence
def find_max_matching_subseq(str1, str2):
    m = len(str1)   # length of str1
    n = len(str2)   # length of str2

    # Initializing the 2-D table
    lookup_table = [[-1 for x in range(n+1)] for y in range(m+1)]

    # Initializing the first row with 0s
    for j in range(n+1):
        lookup_table[0][j] = 0

    # Initializing the first column with 0s
    for i in range(m+1):
        lookup_table[i][0] = 0

    # Iterating to fill the table
    for i in range(1, m+1):           
        # calculate new row (based on previous row i.e. lookup_table)
        for j in range(1, n+1):
            # if characters at this position match, 
            if str1[i-1] == str2[j-1]:    
                # add 1 to the previous diagonal and store it in this diagonal
                lookup_table[i][j] = lookup_table[i-1][j-1] + 1 
            else:
                # If the characters don't match, fill this entry with the max of the
                # left and top elements
                lookup_table[i][j] = max(lookup_table[i][j-1], lookup_table[i-1][j]) 
    return lookup_table[m][n]

def min_del_ins(str1, str2):
  n = find_max_matching_subseq(str1, str2)
  # calculating number of deletions required from str1 to transform it into str2
  deletions = len(str1) - n
  # calculating number of insertions required in str1 to transform it into str2
  insertions = len(str2) - n

  return deletions,insertions

## Edit Distance
'''
Given two strings, str1 and str2, find the minimum edit distance required to convert str1 into str2. Minimum edit distance is the minimum number of insertions, deletions, or substitutions required to transform str1 into str2.
Naive approach: Starting from the last character of both strings, there are two possibilities for every pair of characters being considered:
- The last characters of both the strings match, in which case, the lengths of both the strings are reduced by one, and the function is called recursively on the remaining strings.
- The last characters of both the strings do not match, in which case, all three operations (insertion, deletion, and substitution) are carried out on the last character of the first string.
- The minimum cost for all three operations is computed recursively and returned.
O(3^n) - O(n)
---
Optimized Solution
Top-down solution: the minimum edit distance value is stored in the lookup_table.
If the end characters match, the value returned is stored at lookup_table[m-1][n-1]
If the end characters don’t match, all three operations (insertion, deletion, and substitution) are carried out on the last character of the first string. The minimum cost for all three operations is then computed and stored in the lookup_table[m-1][n-1]
O(n^2) - O(n^2)
Bottom-up solution:  we build the solution bottom-up by dividing our problem into subproblems.

We know that if the first string is empty, we perform the insert operation on all characters of the second string to make both strings similar. Therefore, the minimum number of operations will equal j. Similarly, if the second string is empty, the delete operation is performed on all elements of the first string to make it similar to the second string. So, the minimum number of operations will equal i. Since this is known, we fill the lookup_table accordingly.
If the last characters are the same, we store the returned values from lookup_table[i-1][j-1] in lookup_table[i][j].
For characters that don’t match, the algorithm works by performing all three operations on the last character of the first string and calculating the minimum edit distance required to convert the first string into the second string. This value is then stored in the lookup_table to avoid recalculations.
O(n^2) - O(n^2)
'''
# Naive
def min_edit_dist_rec(str1, str2, m, n):
    
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    if str1[m - 1] == str2[n - 1]:
        return min_edit_dist_rec(str1, str2, m - 1, n - 1)

    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    # adding '1' because every operation has cost of '1'
    return 1 + min(min_edit_dist_rec(str1, str2, m, n - 1),  # Insert
                   min_edit_dist_rec(str1, str2, m - 1, n),  # Remove
                   min_edit_dist_rec(str1, str2, m - 1, n - 1)  # Replace
                   )


def min_edit_dist(str1, str2):
   
    return min_edit_dist_rec(str1, str2, len(str1), len(str2))

# Optimized
# -- Top down
def min_edit_dist_rec(str1, str2, m, n, lookup_table):
   
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # if the recursive call has been
    # called previously, then return
    # the stored value that was calculated
    # previously
    if lookup_table[m - 1][n - 1] != -1:
        return lookup_table[m - 1][n - 1]

    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.

    # Store the returned value at lookup_table[m-1][n-1]
    # considering 1-based indexing
    if str1[m - 1] == str2[n - 1]:
        lookup_table[m - 1][n - 1] = min_edit_dist_rec(str1, str2, m - 1, n - 1, lookup_table)
        return lookup_table[m - 1][n - 1]

    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.

    # Store the returned value at lookup_table[m-1][n-1]
    # considering 1-based indexing
    # adding '1' because every operation has cost of '1'
    lookup_table[m - 1][n - 1] = 1 + min(min_edit_dist_rec(str1, str2, m, n - 1, lookup_table),  # Insert
                                        min_edit_dist_rec(str1, str2, m - 1, n, lookup_table),  # Remove
                                        min_edit_dist_rec(str1, str2, m - 1, n - 1, lookup_table)  # Replace
                                        )
    return lookup_table[m - 1][n - 1]


def min_edit_dist(str1, str2):
   
    # Declare a lookup_table array which stores
    # the answer to recursive calls
    lookup_table = [[-1 for i in range(len(str2))] for i in range(len(str1))]

    return min_edit_dist_rec(str1, str2, len(str1), len(str2), lookup_table)

# -- Bottom up
def min_edit_dist_iterative(str1, str2, m, n):
    
    # Create a table to store results of sub-problems
    lookup_table = [[-1 for i in range(n + 1)] for i in range(m + 1)]

    # Fill lookup_table [][] in bottom up manner
    for i in range(m+1):
        # If second string is empty, only option is to
        # remove all characters of first string
        lookup_table[i][0] = i # Min. operations = i

    for j in range(n+1):
        # If first string is empty, only option is to
        # insert all characters of second string
        lookup_table[0][j] = j # Min. operations = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
           
            # If last characters are same, ignore last char
            # and recur for remaining string
            if str1[i - 1] == str2[j - 1]:
                lookup_table[i][j] = lookup_table[i - 1][j - 1]

            # If the last character is different, consider all
            # possibilities and find the minimum
            # adding '1' because every operation has cost of '1'
            else:
                lookup_table[i][j] = 1 + min(lookup_table[i][j - 1],  # Insert
                                             lookup_table[i - 1][j],  # Remove
                                             lookup_table[i - 1][j - 1])  # Replace

    return lookup_table[m][n]


def min_edit_dist(str1, str2):
    
    return min_edit_dist_iterative(str1, str2, len(str1), len(str2))

## Longest Repeating Subsequence
'''
Given a string, you have to find the length of the longest subsequence that occurs at least twice and respects this constraint: the characters that are re-used in each subsequence should have distinct indexes.
Naive approach: We will start traversing the string from the end by having two indexes. Since both the indexes point at the same character, we will decrease the second index and keep the first one as it is.
On each recursive call, the second index will decrease until the character at the first index gets matched to the second index’s value. After this, both indexes will get decremented and make a recursive call. No match is found and the second index value reaches 0. Now, we will backtrack, and 1 will be returned.
When the first “b” character is matched, we will start the second recursive call. This time the first index will get decremented instead, and the second one will remain the same for all the recursive calls unless “b” is matched again or the first index value becomes 0.
This exact process will be repeated, with the first index changing and the second remaining the same.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n^2) - O(n^2)
Bottom-up solution: We will use a 2-D array to store our results by computing the smaller values first and then finding the larger values using them.
In our lookup_table, indexes p1 and p2 will store the length of the longest repeating subsequence of the substrings str[0 to p1-1] and str[0 to p2-1].
O(n^2) - O(n^2)
'''
# Naive
def find_LRS(str):
    return find_LRS_recursive(str, len(str), len(str))


def find_LRS_recursive(str, p1, p2):
    # Base case if any one index has reached end
    if (p1 == 0) or (p2 == 0):
        return 0

    # Characters are same but indexes are different
    if (str[p1 - 1] == str[p2 - 1]) and (p1 != p2):
        return find_LRS_recursive(str, p1 - 1, p2 - 1) + 1

    # Check if the characters at both indexes don't match
    return max(
        find_LRS_recursive(str, p1, p2 - 1), find_LRS_recursive(str, p1 - 1, p2)
    )

# Optimized
# -- Top down
def find_LRS(str):
    lookup_table = {}
    return find_LRS_memoization(str, len(str), len(str), lookup_table)


def find_LRS_memoization(str, p1, p2, lookup_table):
    # Base case if any one index has reached end
    if (p1 == 0) or (p2 == 0):
        return 0

    # Creating a key to store in map
    key = p1, ",", p2
    
    # Checking if unique pair not in lookup table
    if key not in lookup_table:
        # Characters are same but indexes are different
        if (str[p1 - 1] == str[p2 - 1]) and (p1 != p2):
            lookup_table[key] = (
                find_LRS_memoization(str, p1 - 1, p2 - 1, lookup_table) + 1
            )
        # Check if the characters at both indexes don't match
        else:
            lookup_table[key] = max(
                find_LRS_memoization(str, p1, p2 - 1, lookup_table),
                find_LRS_memoization(str, p1 - 1, p2, lookup_table),
            )

    # Returning value of the key from the map
    return lookup_table[key]

# -- Bottom up
def find_LRS(str):
    size = len(str)
    # Create a table to hold intermediate values
    lookup_table = [[0 for x in range(size + 1)] for y in range((size + 1))]

    # Starting from second row, filling the lookup table bottom-up wise
    for i in range(1, size + 1):
        for k in range(1, size + 1):
            # Characters are same but indexes are different
            if str[i - 1] == str[k - 1] and i != k:
                lookup_table[i][k] = lookup_table[i - 1][k - 1] + 1
            # Check if the characters at both indexes don't match
            else:
                lookup_table[i][k] = max(lookup_table[i - 1][k], lookup_table[i][k - 1])

    # Returning the longest repeating subsequence length
    return lookup_table[size][size]

## Distinct Subsequence Pattern Matching
'''
Given two strings, str1 and str2, return the number of times str2 appears in str1 as a subsequence.
Naive approach: we can set up the base cases and recursive cases of our algorithm.
Base cases:
If we have reached the end of str2, that means we have found a subsequence in str1 that contains str2. Therefore, we return 1.
Similarly, if we have reached the end of str1, that means we have exhausted all the characters in str1 before reaching the end of str2 (else, we’d have already reached the first base case) and no subsequence containing all the characters in str2 was found. In this case, we return 0.
Recursive cases:
If both the characters, str1[i1] and str2[i2], are same, we make two recursive calls to the function:
- One where we include the current match, that is, we increment both i1 and i2 by 1.
- The second, where we ignore the current match, that is, we only increment i1.
If the two characters, str1[i1] and str2[i2], do not match, we only increment the i1 pointer by 1.
O(2^(m+n)) - O(m+n)
---
Optimized solution
Top-down solution: O(mn) - O(mn)
Bottom-up solution: we create a lookup table of size (m+1)×(n+1) and fill the last column with ones and the last row with zeros. The additional row and column represent empty substrings of str1 and str2 and correspond to the base cases mentioned in the naive approach. These are the smallest subproblems that we need to solve.
We iterate over the remaining lookup table, starting from the bottom-right corner and moving to the left to fill up the last row, and then the second-last row, and so on, until we reach the top-left corner of the table. To do this, we iterate backwards through str1, starting i1 from m−1 and moving towards 0, and, for each i1, we iterate backwards through str2 using i2, going from n−1 to 0. This way, we build upon the solutions of the smaller subproblems in order to compute the solution of the overall problem.
In each iteration, we calculate the number of distinct subsequences in str1[i1:m] that contain str2[i2:n] using the following conditional logic:
If str1[i1] == str2[i2], we add the number of subsequences stored in lookup_table[i1+1][i2+1] (the case where we include the match) and lookup_table[i1+1][i2] (the case where we ignore the match), and store the result in lookup_table[i1][i2].
If str1[i1] != str2[i2], the number of subsequences in str1[i1:m] containing str2[i2:n] will be lookup_table[i1+1][i2]. Therefore, this result is stored in lookup_table[i1][i2].
Finally, the number of subsequences in str1 that contain str2 will be stored in lookup_table[0][0].
O(mn) - O(mn)
'''
# Naive
# helper recursive function
def number_of_subsequences_rec(str1, str2, m, n, i1, i2):
    # if we have reached the end of str1, return 1
    if i2 == n:
        return 1
    
    # if we have reached the end of str2, return 0
    if i1 == m:
        return 0
    
    # initializing result variable to store the number of subsequences
    result = 0

    # if both the characters are same
    if str1[i1] == str2[i2]:
        result += number_of_subsequences_rec(str1, str2, m, n, i1 + 1, i2 + 1)
        result += number_of_subsequences_rec(str1, str2, m, n, i1 + 1, i2)     # ignoring this match
    # if the two characters are different
    else:
        result += number_of_subsequences_rec(str1, str2, m, n, i1 + 1, i2)
    
    # return the number of subsequences
    return result

def number_of_subsequences(str1, str2):
    # initializing variables
    m = len(str1)
    n = len(str2)

    # calling the helper recursive function
    num_subsequences = number_of_subsequences_rec(str1, str2, m, n, 0, 0)
    
    # returning the results
    return num_subsequences

# Optimized
# -- Top down
# helper recursive function
def number_of_subsequences_rec(str1, str2, m, n, i1, i2, lookup_table):
    # if we have reached the end of str1, return 1
    if i2 == n:
        return 1
    
    # if we have reached the end of str2, return 0
    if i1 == m:
        return 0
    
    # if the result is not present in the lookup table
    if lookup_table[i1][i2] == -1:
        # if both the characters are same
        if str1[i1] == str2[i2]:
            lookup_table[i1][i2] = number_of_subsequences_rec(str1, str2, m, n, i1 + 1, i2 + 1, lookup_table) 
            lookup_table[i1][i2] += number_of_subsequences_rec(str1, str2, m, n, i1 + 1, i2, lookup_table)     # ignoring this match
        # if the two characters are different
        else:
            lookup_table[i1][i2] = number_of_subsequences_rec(str1, str2, m, n, i1 + 1, i2, lookup_table)
    
    # return the result stored in the lookup table
    return lookup_table[i1][i2]

def number_of_subsequences(str1, str2):
    # initializing variable
    m = len(str1)
    n = len(str2)

    # initializing lookup table to store the results of recursive calls
    lookup_table = [[-1 for x in range(0, len(str2))] for x in range(0, len(str1))]
    
    # call the recursive helper function
    num_subsequences = number_of_subsequences_rec(str1, str2, m, n, 0, 0, lookup_table)
    
    # return the results
    return num_subsequences

# -- Bottom up
# function to calculate the number of subsequences
def number_of_subsequences(str1, str2):
    # initializing variables
    m = len(str1)
    n = len(str2)
    lookup_table = [[0 for x in range(0, n + 1)] for x in range(0, m + 1)]

    # filling the last row with 0s
    for i in range(0, n + 1):
        lookup_table[m][i] = 0
    
    # filling the last column with 1s
    for i in range(0, m + 1):
        lookup_table[i][n] = 1
    
    # iterating over the lookup table starting from m-1 and n-1
    for i1 in range(m - 1, -1, -1):
        for i2 in range(n - 1, -1, -1):
            # if both the characters are same
            if str1[i1] == str2[i2]:
                lookup_table[i1][i2] += lookup_table[i1 + 1][i2 + 1] + lookup_table[i1 + 1][i2]
            # if the two characters are different
            else:
                lookup_table[i1][i2] = lookup_table[i1 + 1][i2]
    
    # returning the result stored in lookup_table[0][0]
    return lookup_table[0][0]

## Maybe You Should Talk to Someone
'''
Given strings s1, s2, and s3, find whether an interleaving of s1 and s2 forms s3.
Naive approach: The first step is to verify if the length of s3 is equal to the sum of the lengths of s1 and s2. If it isn’t, return FALSE as the output. However, that isn’t the case for the given inputs. Therefore, we can traverse these strings and engage in the matching process. While traversing, we keep track of the indexes we are at in all three strings, allowing us to check the order in which these characters appear.
We have two options at any step:
- If the letter at s1 index matches with the letter at s3 index, we consume both, that is, we move s1 index as well as s3 index one step forward. The matching process continues recursively for the portions of s1, s2, and s3 that we have not yet considered.
- If the letter at s2 index matches with the letter at s3 index, we consume both, that is, we move s2 index as well as s3 index one step forward. The matching process continues recursively for the portions of s1, s2, and s3 that we have not yet considered.
This process allows us to check all possible combinations of interleaving.
O(2^(s+t)) - O(s+t)
---
Optimized solution
Top-down solution: O(st) - O(st)
Bottom-up solution: Check solution O(st) - O(st)
'''
# Naive
def find_strings_interleaving_recursive(s1, s2, s3, s1_index, s2_index, s3_index):

    # If we have reached the end of all the strings
    if s1_index == len(s1) and s2_index == len(s2) and s3_index == len(s3):
        return True

    # If we have reached the end of 's3' but 's1' or 's2' still has some characters left
    if s3_index == len(s3):
        return False

    # The two decisions we can make at every instance are initially set to false
    d1 = False
    d2 = False

    # If s1 index and s3 index are pointing to the same character
    if s1_index < len(s1) and s1[s1_index] == s3[s3_index]:
        d1 = find_strings_interleaving_recursive(
            s1, s2, s3, s1_index + 1, s2_index, s3_index + 1)

    # If s2 index and s3 index are pointing to the same character
    if s2_index < len(s2) and s2[s2_index] == s3[s3_index]:
        d2 = find_strings_interleaving_recursive(
            s1, s2, s3, s1_index, s2_index + 1, s3_index + 1)

    # If either of these conditions is true, it means that s3 was indeed a product of
    # interleaving s1 and s2.
    return d1 or d2


# The main function that initiates a recursive call to the helper function
def is_interleaving(s1, s2, s3):
    return find_strings_interleaving_recursive(s1, s2, s3, 0, 0, 0)

# Optimized
# -- Top down
def find_strings_interleaving_memoization(s1, s2, s3, s1_index, s2_index, s3_index, table):

    # If we have reached the end of all the strings
    if s1_index == len(s1) and s2_index == len(s2) and s3_index == len(s3):
        return True

    # If we have reached the end of 's3' but 's1' or 's2' still has some characters left
    if s3_index == len(s3):
        return False

    # Setting up the key for this subproblem
    sub_problem = str(s1_index) + "/" + str(s2_index) + "/" + str(s3_index)
    if sub_problem not in table: # Checking if the sub problem is already solved
      d1 = False
      d2 = False
      # If s1 index and s3 index are pointing to the same character
      if s1_index < len(s1) and s1[s1_index] == s3[s3_index]:
          d1 = find_strings_interleaving_memoization(
              s1, s2, s3, s1_index + 1, s2_index, s3_index + 1, table)

      # If s2 index and s3 index are pointing to the same character
      if s2_index < len(s2) and s2[s2_index] == s3[s3_index]:
          d2 = find_strings_interleaving_memoization(
              s1, s2, s3, s1_index, s2_index + 1, s3_index + 1, table)

      table[sub_problem] = d1 or d2
    # If either of these starting decisions result in us verifying that s3 was indeed a product of
    # interleaving s1 and s2, than we will get True, otherwise False
    return table.get(sub_problem)


# The main function that initiates a recursive call to the helper function, however note that
# this time we also add an empty hash table as one of the parameters. 
def is_interleaving(s1, s2, s3):
    return find_strings_interleaving_memoization(s1, s2, s3, 0, 0, 0, {})

# -- Bottom up
def is_interleaving(s1, s2, s3):
    # For the empty pattern, we have one matching
    if len(s1) + len(s2) != len(s3):
        return False

    # Create a table with an extra row and column to separate the base cases.
    lookup_table = [[False for i in range(len(s2) + 1)] for i in range(len(s1) + 1)]

    for s1_index in range(len(s1) + 1):
        for s2_index in range(len(s2) + 1):

            # If 's1' and 's2' are empty, then 's3' must have been empty too.
            if s1_index == 0 and s2_index == 0:
                lookup_table[s1_index][s2_index] = True
            # Checking the interleaving with 's2' only
            elif s1_index == 0 and s2[s2_index - 1] == s3[s1_index + s2_index - 1]:
                lookup_table[s1_index][s2_index] = lookup_table[s1_index][s2_index - 1]
            # Checking the interleaving with 's1' only
            elif s2_index == 0 and s1[s1_index - 1] == s3[s1_index + s2_index - 1]:
                lookup_table[s1_index][s2_index] = lookup_table[s1_index - 1][s2_index]
            else:
                # If the letter of 's1' and 's3' match, we take whatever is matched till s1_index-1
                if s1_index > 0 and s1[s1_index - 1] == s3[s1_index + s2_index - 1]:
                    lookup_table[s1_index][s2_index] = lookup_table[s1_index - 1][s2_index]

                # If the letter of 's2' and 's3' match, we take whatever is matched till s2_index-1 too
                # note the '|=', this is required when we have common letters
                if s2_index > 0 and s2[s2_index - 1] == s3[s1_index + s2_index - 1]:
                    lookup_table[s1_index][s2_index] |= lookup_table[s1_index][s2_index - 1]

    return lookup_table[len(s1)][len(s2)]

## Word Break II
'''
Given a string s and a dictionary of strings word_dict, add spaces to s to break it up into a sequence of valid words from word_dict. We are required to return all possible sequences of words (sentences). The order in which the sentences are listed is not significant.
Naive approach: steps to implement the algorithm:
If a string s is empty, there’ll be no sentences that can be formed. So, we return the empty list.
If s is not empty, we’ll iterate every word of the dictionary and check if s starts with the current dictionary word or not.
- If it doesn’t start with the current dictionary word, we move on to the next word.
- If it does, we’ll keep that word as a prefix and recursively perform the same steps for the rest of the string.
- We’ll then concatenate the prefix and the result of the suffix computed by a recursive call.
- If the length of the word and the length of the remaining string are the same, we append the remaining string to our sentence.
O(w*2^n) - O(n+w)
---
Optimized solution 
Top-down solution: algorithm for the implementation:
The word_break function takes in the string s and the list of words called word_dict. This function then calls a recursive word_break_rec function.
The recursive function takes in the string s, the word_dict list, and an empty hash map. We use this hash map to store results for each substring.
The recursive function’s base case is when the s string is empty; this returns an empty list. Note that it’s actually an empty list of lists because that’s the return type of this function.
In the recursive function, we run an iteration over all the prefixes of the query. If a prefix matches a word in the list, we recursively invoke the function on the postfix.
At the end of the iteration, we store the results in the hash map called result, with each valid postfix string as the key and the list of words it can be broken up into, as the value. For instance, for the postfix “cookbook”, the corresponding hash map entry would be “cook book”.
We return the value from result that corresponds to the original query string.
O(n^2) - O(n)
Bottom-up solution: algorithm for the implementation:
Initialize an empty lookup table, dp_solutions. This table will be used to store the solutions to previously solved subproblems. The values of this table will be empty lists at the start.
As the base case, we set the first index of dp_solutions, that is, the solution to the problem "sentences made from substring of length 0". We set this to be a list with one element, an empty string.
Iterate over the input string, breaking it down into the possible substrings by including a single character, one at a time. Additionally, initialize an array called temp that will store the result of the current substring being checked.
For all possible prefixes of the current substring, check if the prefix exists in the given dictionary. If it does, we know that the prefix is a valid word from the dictionary and can be used as part of the solution.
If the prefix is present in the dictionary, check if any part of the current substring already exists in the dp_solutions array. If it does, retrieve that part from the dp_solutions array and append the prefix to it, with a space character to separate them, and add this intermediate result to the temp array.
For each substring in the string s, repeat the process. After each iteration, save the results in the dp_solutions table.
Return the value at the last index of the dp_solutions table as this index contains all possible sentences formed from a complete string s.
O(n^2) - O(n)
'''
# Naive9
def word_break(s, word_dict):
    # Calling the word_break_rec function
    return word_break_rec(s, word_dict)
    
# Helper Function that breaks down the string into words from subs
def word_break_rec(s, dictionary):
    # If s is empty string
    if not s:
        return []
    
    res = []
    for word in dictionary:
        # Verifying if s can be broken down further
        if not s.startswith(word):
            continue
        if len(word) == len(s):
            res.append(s)
        else:
            result_of_the_rest = word_break_rec(s[len(word):], dictionary)
            for item in result_of_the_rest:
                item = word + ' ' + item
                res.append(item)
    return res

# Optimized
# -- Top down
def word_break(s, word_dict):
    # Calling the helper function
    return word_break_rec(s, word_dict, {})
    
# Helper Function that breaks down the string into words from subs
def word_break_rec(s, dictionary, result):
    # If s is empty string
    if not s:
        return []
    
    if s in result:
        return result[s]
    
    res = []
    for word in dictionary:
        # Verifying if s can be broken down further
        if not s.startswith(word):
            continue
        if len(word) == len(s):
            res.append(s)
        else:
            result_of_the_rest = word_break_rec(s[len(word):], dictionary, result)
            for item in result_of_the_rest:
                item = word + ' ' + item
                res.append(item)
    result[s] = res
    return res

# -- Bottom up
def word_break(s, word_dict):

    # Initializing a table of size s.length + 1
    dp_solutions = [[]] * (len(s)+1)

    # Setting base case
    dp_solutions[0] = [""]

    # For each substring in the input string, repeat the process.
    for i in range(1, len(s)+1):

        # An array to store the results of the current substring being checked.
        temp = []

        # Iterate over the current substring and break it down into all possible prefixes.
        for j in range(0, i):
            prefix = s[j:i]
            
            # Check if the current prefix exists in word_dict. If it does, we know that it is a valid word
            # and can be used as part of the solution.
            if prefix in word_dict:
                
                # Check if any part of the current substring already exists in the dp_solutions array.
                for substrings in dp_solutions[j]:
                    # Merge the prefix with the already calculated results
                    temp.append((substrings + " " + prefix).strip())

        dp_solutions[i] = temp

    # Returning all the sentences formed using a complete string s.
    return dp_solutions[len(s)]

## Longest Increasing Subsequence
'''
The Longest Increasing Subsequence (LIS) is the longest subsequence from a given array in which the subsequence elements are sorted in strictly increasing order. Given an integer array, nums, find the length of the LIS in this array.
Naive approach: all the ideas that are needed to find the solution to this problem:
Each element of the array is either a part of an increasing subsequence or it is not.
To be part of an increasing subsequence, the current element needs to be greater than the previous element included in the subsequence.
To find the longest increasing subsequence, we need to move from left to right, exhaustively generating all increasing subsequences, while keeping track of the longest one found at any step.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n^2) - O(n^2)
Bottom-up solution: O(n^2) - O(n^2)
'''
# Naive
def LIS_length(nums):
  return LIS_length_rec(nums, 0, -1) 

def LIS_length_rec(nums, curr, prev):
  # base case
  # if 'curr' reaches the end of the array, return 0
  if curr == len(nums):
    return 0

  # solve the first subproblem
  length = LIS_length_rec(nums, curr+1, prev)  # calculate the LIS length from 'curr+1', skipping the current element
  
  # solve the second subproblem
  if prev < 0 or nums[prev] < nums[curr]: # if the current element is greater than the previous one in the subsequence
    length = max(length, 1+LIS_length_rec(nums, curr+1, curr))  # calculate the LIS length from 'curr+1', including the current element
  return length

# Optimized
# -- Top down
def LIS_length(nums):
  length = len(nums)
  # we created a table here
  dp = [[-1]*(length+2) for i in range(length+1)]
  return LIS_length_rec(nums, 0, -1, dp)

def LIS_length_rec(nums, curr, prev, dp):
  # base case
  # if 'curr' reaches the end of the array, return 0
  if curr == len(nums):
    return 0

  # if subproblem has already been solved, use the stored result
  if dp[curr][prev+1] != -1:
    return dp[curr][prev+1]
  
  # solve the first subproblem 
  length = LIS_length_rec(nums, curr+1, prev, dp)
  
  # solve the second subproblem
  if prev < 0 or nums[prev] < nums[curr]:
    length = max(length, 1+LIS_length_rec(nums, curr+1, curr, dp))
  
  dp[curr][prev+1] = length # store the result of the current subproblem
  return dp[curr][prev+1]

# -- Bottom up
def LIS_length(nums):
    size = len(nums)
    # we created a table here
    dp = [[0]*(size+1) for i in range(size+1)]

    for curr in range(size-1, -1, -1):
        for prev in range(curr-1, -2, -1):
            length = dp[curr+1][prev+1]
            # if 'prev' is negative or previous value is less than the next value
            # we will take it
            if prev < 0 or nums[prev] < nums[curr]:
                length = max(length, 1+dp[curr+1][curr+1])
            dp[curr][prev+1] = length
    return dp[0][0]

## Minimum Deletions to Make a Sequence Sorted
'''
Given an integer array nums, the task is to remove or delete the minimum number of elements from the array so that the remaining elements form a strictly increasing sequence. This is very similar to the Longest Increasing Subsequence (LIS) problem because elements other than the longest increasing subsequence should be removed to make the sequence sorted.
Naive approach: steps of the solution:
Calculate the length of the longest increasing subsequence.
Subtract that from the total length of the array.
To compute the length of the LIS:
Each element of the array is either a part of an increasing subsequence or it is not.
To be part of an increasing subsequence, the current element needs to be greater than the previous element included in the subsequence.
To find the longest increasing subsequence, we need to move from left to right, exhaustively generating all increasing subsequences, while keeping track of the longest one found at any step.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n^2) - O(n^2)
Bottom-up solution: O(n^2) - O(n^2)
'''
# Naive
def min_deletions(nums):
  # subtract length of longest increasing subsequence from total length
  minimum_deletions = len(nums) - LIS_length(nums, 0, -1) 
  return minimum_deletions

def LIS_length(nums, curr, prev):
  # base case
  # if 'curr' reaches the end of the array, return 0
  if curr == len(nums):
    return 0

  # solve the first subproblem
  length = LIS_length(nums, curr+1, prev)  # calculate the LIS length from 'curr+1', skipping the current element
  
  # solve the second subproblem
  if prev < 0 or nums[prev] < nums[curr]: # if the current element is greater than the previous one in the subsequence
    length = max(length, 1+LIS_length(nums, curr+1, curr))  # calculate the LIS length from 'curr+1', including the current element
  return length

# Optimized
# -- Top down
def min_deletions(nums):
  length = len(nums)
  # we created a table here
  dp = [[-1]*(length+2) for i in range(length+1)]
  # subtract length of longest increasing subsequence from total length
  minimum_deletions = length - LIS_length(nums, 0, -1, dp)
  return minimum_deletions

def LIS_length(nums, curr, prev, dp):
  # base case
  # if 'curr' reaches the end of the array, return 0
  if curr == len(nums):
    return 0

  # if subproblem has already been solved, use the stored result
  if dp[curr][prev+1] != -1:
    return dp[curr][prev+1]
  
  # solve the first subproblem 
  length = LIS_length(nums, curr+1, prev, dp)
  
  # solve the second subproblem
  if prev < 0 or nums[prev] < nums[curr]:
    length = max(length, 1+LIS_length(nums, curr+1, curr, dp))
  
  dp[curr][prev+1] = length # store the result of the current subproblem
  return dp[curr][prev+1]

# -- Bottom up
def min_deletions(nums):
    size = len(nums)
    # we created a table here
    dp = [[0]*(size+1) for i in range(size+1)]

    for curr in range(size-1, -1, -1):
        for prev in range(curr-1, -2, -1):
            length = dp[curr+1][prev+1]
            # if 'prev' is negative or previous value is less than the next value
            # we will take it
            if prev < 0 or nums[prev] < nums[curr]:
                length = max(length, 1+dp[curr+1][curr+1])
            dp[curr][prev+1] = length
    # subtract length of longest increasing subsequence from total length
    minimum_deletions = size - dp[0][0]
    return minimum_deletions

## Maximum Sum Increasing Subsequence
'''
Given an array of integers nums, identify the increasing subsequence whose sum is the highest among all increasing subsequences. Return the sum. Note that an increasing subsequence of an array is defined as a subsequence whose elements are sorted in strictly increasing order.
Naive approach: all the ideas that are needed to find the solution to this problem:
Each element of the array is either a part of an increasing subsequence or it is not.
To be part of an increasing subsequence, the current element needs to be greater than the previous element included in the subsequence.
To find the maximum sum increasing subsequence, we need to move from left to right, exhaustively generating all increasing subsequences, while keeping track of the one with the greatest sum found so far.
O(2^n) - O(n)
---
Optimized solution
Top-down solution: O(n^2) - O(n^2)
Bottom-up solution: O(n^2) - O(n^2)
'''
# Naive
def MSIS_length(nums):
  sum = 0
  return MSIS_length_rec(nums, 0, -1, sum) 

def MSIS_length_rec(nums, curr, prev, sum):
  # base case
  # if 'curr' reaches the end of the array, return the current sum
  if curr == len(nums):
    return sum

  # solve the first subproblem
  max_sum = MSIS_length_rec(nums, curr+1, prev, sum)  # calculate the MSIS from 'curr+1', skipping the current element

  # solve the second subproblem
  if prev < 0 or nums[prev] < nums[curr]: # if the current element is greater than the previous one in the subsequence
    max_sum = max(max_sum, MSIS_length_rec(nums, curr+1, curr, sum+nums[curr])) # calculate the MSIS length from 'curr+1', including the current element
  return max_sum

# Optimized
# -- Top down
def MSIS_length(nums):
  length = len(nums)
  sum = 0
  # we created a table here
  dp = [[-1]*(length+2) for i in range(length+1)]
  return MSIS_length_rec(nums, 0, -1, dp, sum)

def MSIS_length_rec(nums, curr, prev, dp, sum):
  # base case
  # if 'curr' reaches the end of the array,  return the current sum
  if curr == len(nums):
    return sum
  
  # if subproblem has alredy been solved, use the stored result
  if dp[curr][prev+1] != -1:
    return dp[curr][prev+1]

  # solve the first subproblem
  max_sum = MSIS_length_rec(nums, curr+1, prev, dp, sum)
  
  # solve the second subproblem
  if prev < 0 or nums[prev] < nums[curr]:
    max_sum = max(max_sum, sum+MSIS_length_rec(nums, curr+1, curr, dp, nums[curr]))
  
  dp[curr][prev+1] = max_sum  # store the result of the current subproblem
  return dp[curr][prev+1]

# -- Bottom up
def MSIS_length(nums):
    size = len(nums)
    # we created a table here
    dp = [[0]*(size+1) for i in range(size+1)]

    for curr in range(size-1, -1, -1):
        for prev in range(curr-1, -2, -1):
            length = dp[curr+1][prev+1]
            # if 'prev' is negative or previous value is less than the next value
            # we will take it
            if prev < 0 or nums[prev] < nums[curr]:
                length = max(length, nums[curr]+dp[curr+1][curr+1])
            dp[curr][prev+1] = length
    return dp[0][0]

