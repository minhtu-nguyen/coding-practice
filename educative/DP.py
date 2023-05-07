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

