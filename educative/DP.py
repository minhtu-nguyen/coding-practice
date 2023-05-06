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