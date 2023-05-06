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