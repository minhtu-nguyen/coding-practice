### *** DP
## Coin Change
'''
COINS in different denominations of 'k', c1, c2...Ck ', the number of each coin is unlimited, and then I give you a total amount 'amount', and I ask you  at least  how many COINS are needed to scrape up this amount, if it is impossible, the algorithm returns -1.
For example 'k = 3', face value 1,2,5, total amount 'amount = 11'.So you have to have at least 3 COINS, so 11 is equal to 5 plus 5 plus 1.
'''
# Brute force O(kn^k)
def coinChange(coins: List[int], amount: int):

    def dp(n):
        # base case
        if n == 0: return 0
        if n < 0: return -1
        # to minimize it is to initialize it to infinity
        res = float('INF')
        for coin in coins:
            subproblem = dp(n - coin)
            # No solution to subproblem, skip
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)

        return res if res != float('INF') else -1

    return dp(amount)

# Memo O(kn)
def coinChange(coins: List[int], amount: int):
    # memo
    memo = dict()
    def dp(n):
        # Check the memo to avoid double counting
        if n in memo: return memo[n]

        if n == 0: return 0
        if n < 0: return -1
        res = float('INF')
        for coin in coins:
            subproblem = dp(n - coin)
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)

        # note on memo
        memo[n] = res if res != float('INF') else -1
        return memo[n]

    return dp(amount)

# Iterative - Bottom up 
int coinChange(vector<int>& coins, int amount) {
    // The array size is amount + 1 and the initial value is also amount + 1
    vector<int> dp(amount + 1, amount + 1);
    // base case
    dp[0] = 0;
    for (int i = 0; i < dp.size(); i++) {
        // The inner for is finding the minimum of + 1 for all subproblems
        for (int coin : coins) {
            // No solution to subproblem, skip
            if (i - coin < 0) continue;
            dp[i] = min(dp[i], 1 + dp[i - coin]);
        }
    }
    return (dp[amount] == amount + 1) ? -1 : dp[amount];
}

## Traversal order of the dp array
# Forward
int[][] dp = new int[m][n];
for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
        # Calculate dp[i][j]
# Backward
for (int i = m - 1; i >= 0; i--)
    for (int j = n - 1; j >= 0; j--)
        // Calculate dp[i][j]

# Diagonally
// Traverse the array diagonally
for (int l = 2; l <= n; l++) {
    for (int i = 0; i <= n - l; i++) {
        int j = l + i - 1;
        // Calculate dp[i][j]
    }
}

## Longest Common Subsequence
# Brute force
def longestCommonSubsequence(str1, str2) -> int:
    def dp(i, j):
        #  base case
        if i == -1 or j == -1:
            return 0
        if str1[i] == str2[j]:
            # found a character belongs to lcs, keep finding
            return dp(i - 1, j - 1) + 1
        else:
            # it's up to the character which can make lcs longer
            return max(dp(i-1, j), dp(i, j-1))

    # i and j became the indexes of the final character in lcs
    return dp(len(str1)-1, len(str2)-1)

# DP Memo
def longestCommonSubsequence(str1, str2) -> int:
    m, n = len(str1), len(str2)
    # construct DP table and base case
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # state transition
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # found a character in lcs
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[-1][-1]

## Edit Distance
'''
For each pair of characters, s1[i] and s2[j], there are four operations:
if s1[i] == s2[j]:
    skip
    i, j move forward
else:
    chose：
        insert
        delete
        replace
'''
def minDistance(s1, s2) -> int:

    def dp(i, j):
        # base case
        if i == -1: return j + 1
        if j == -1: return i + 1

        if s1[i] == s2[j]:
            return dp(i - 1, j - 1)  # skip
        # explanation：
        # already the same, no need of any operations
        # the least editing distance of s1[0..i] and s2[0..j] equals
        # the least distance of s1[0..i-1] and s2[0..j-1]
        # It means that dp(i, j) equals dp(i-1, j-1)
        else:
            return min(
                dp(i, j - 1) + 1,    # insert
                # explanation：
                # Directly insert the same character as s2[j] at s1[i]
                # then s2[j] is matched，move forward j，and continue comparing with i
                # Don't forget to add one to the number of operations
                dp(i - 1, j) + 1,    # delete
                # explanation：
                # Directly delete s[i]
                # move i forward，continue comparing with j
                # add one to the number of operations
                dp(i - 1, j - 1) + 1 # replace
                # explanation：
                # Directly replace s1[i] with s2[j], then they are matched
                # move forward i，j and continue comparing
                # add one to the number of operations
            )

    # i，j initialize to the last index
    return dp(len(s1) - 1, len(s2) - 1)

# Memo
def minDistance(s1, s2) -> int:

    memo = dict() # memo
    def dp(i, j):
        if (i, j) in memo: 
            return memo[(i, j)]
        ...

        if s1[i] == s2[j]:
            memo[(i, j)] = ...  
        else:
            memo[(i, j)] = ...
        return memo[(i, j)]

    return dp(len(s1) - 1, len(s2) - 1)

# Bottom up
'''
int minDistance(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    int[][] dp = new int[m + 1][n + 1];
    // base case 
    for (int i = 1; i <= m; i++)
        dp[i][0] = i;
    for (int j = 1; j <= n; j++)
        dp[0][j] = j;
    // from the bottom up
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (s1.charAt(i-1) == s2.charAt(j-1))
                dp[i][j] = dp[i - 1][j - 1];
            else               
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i-1][j-1] + 1
                );
    // store the least editing distance of s1 and s2
    return dp[m][n];
}

int min(int a, int b, int c) {
    return Math.min(a, Math.min(b, c));
}
'''

## Super Egg
'''
Suppose there are several floors in a high building and several eggs in your hands, lets calculate the minimum number of attempts and find out the floor where the eggs just won’t be broken. 
https://labuladong.gitbook.io/algo-en/i.-dynamic-programming/throwingeggsinhighbuildings
'''
# Brute force - Recursive
def dp(K, N):
    for i in range(1, N + 1):
        # base case
        if K == 1: return N
        if N == 0: return 0
        # Minimum number of eggs throwing in the worst case
        res = min(res, 
                  max( 
                        dp(K - 1, i - 1), # broken
                        dp(K, N - i)      # not broken
                     ) + 1 # throw once on the i-th floor
                 )
    return res
# Memo
def superEggDrop(self, K: int, N: int) -> int:

    memo = dict()
    def dp(K, N):
        if K == 1: return N
        if N == 0: return 0
        if (K, N) in memo:
            return memo[(K, N)]

        # for 1 <= i <= N:
        #     res = min(res, 
        #             max( 
    #                     dp(K - 1, i - 1), 
    #                     dp(K, N - i)      
        #                 ) + 1 
        #             )

        res = float('INF')
        # use binary search instead of for loop(linear search)
        lo, hi = 1, N
        while lo <= hi:
            mid = (lo + hi) // 2
            broken = dp(K - 1, mid - 1) # broken
            not_broken = dp(K, N - mid) # not broken
            # res = min(max(broken，not broken) + 1)
            if broken > not_broken:
                hi = mid - 1
                res = min(res, broken + 1)
            else:
                lo = mid + 1
                res = min(res, not_broken + 1)

        memo[(K, N)] = res
        return res

    return dp(K, N)

## Subsequence Problem
'''
 the subsequence problem itself is more difficult than those for substring and subarray, since the former needs to deal with discontinuous sequence, while the latter two are continuous.
'''
# one-dimensional DP array
'''
int n = array.length;
int[] dp = new int[n];

for (int i = 1; i < n; i++) {
    for (int j = 0; j < i; j++) {
        dp[i] = max|min(dp[i], dp[j] + ...)
    }
}
"the Longest Increasing Subsequence (LIS)". The definition of DP array in this case is as below:
We define dp[i] as the length of the required subsequence (the longest increasing subsequence) within the subarray array [0..i].
'''
# two-dimensional DP array
'''
int n = arr.length;
int[][] dp = new dp[n][n];

for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        if (arr[i] == arr[j]) 
            dp[i][j] = dp[i][j] + ...
        else
            dp[i][j] = max|min(...)
    }
}
The definition of the DP array in this strategy is further divided into two cases: "Only one string is involved" and "Two strings are involved".
a) In the case where two strings are involved (e.g. LCS), the definition of DP array is as follows:
We define dp[i][j] as the length of the required subsequence (longest common subsequence) within the subarray arr1[0..i] and the subarray arr2[0..j].
b) In the case where only one string is involved (such as the Longest Palindrome Subsequence (LPS) which will be discussed in this article later), the definition of DP array is as follows:
We define dp[i][j] as the length of the required subsequence (the longest palindrome subsequence) within the subarray array [i..j].
For the first case, you can refer to these two articles: "Editing distance", "Common Subsequence".
'''
## Longest Palindrome Subsequence
'''
int longestPalindromeSubseq(string s) {
    int n = s.size();
    // DP arrays are all initialized to 0
    vector<vector<int>> dp(n, vector<int>(n, 0));
    // base case
    for (int i = 0; i < n; i++)
        dp[i][i] = 1;
    // Reverse traversal to ensure correct state transition
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            // State transition equation
            if (s[i] == s[j])
                dp[i][j] = dp[i + 1][j - 1] + 2;
            else
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
        }
    }
    // return the length of LPS
    return dp[0][n - 1];
}
'''

## Game Problems
'''
class Pair {
    int fir, sec;
    Pair(int fir, int sec) {
        this.fir = fir;
        this.sec = sec;
    }
}

/* Returns the difference between the last first hand and last hand */
int stoneGame(int[] piles) {
    int n = piles.length;
    //Initializes the dp array
    Pair[][] dp = new Pair[n][n];
    for (int i = 0; i < n; i++) 
        for (int j = i; j < n; j++)
            dp[i][j] = new Pair(0, 0);
    // base case
    for (int i = 0; i < n; i++) {
        dp[i][i].fir = piles[i];
        dp[i][i].sec = 0;
    }
    // traverse the array diagonally
    for (int l = 2; l <= n; l++) {
        for (int i = 0; i <= n - l; i++) {
            int j = l + i - 1;
            // The first hand select the left- or right-most pile.
            int left = piles[i] + dp[i+1][j].sec;
            int right = piles[j] + dp[i][j-1].sec;
            // Refer to the state transition equation.
            if (left > right) {
                dp[i][j].fir = left;
                dp[i][j].sec = dp[i+1][j].fir;
            } else {
                dp[i][j].fir = right;
                dp[i][j].sec = dp[i][j-1].fir;
            }
        }
    }
    Pair res = dp[0][n-1];
    return res.fir - res.sec;
}
'''

## Interval Scheduling - Greedy
'''
Given a series of closed intervals [start, end] , you should design an algorithm to compute the number of maximum subsets without any overlapping.
For example,intvs = [[1,3], [2,4], [3,6]], the interval set have 2 subsets without any overlapping at most, [[1,3], [3,6]] , so your algorithm should return 2 as the result. Note that intervals with the same border doesn't meet the condition.

public int intervalSchedule(int[][] intvs) {
    if (intvs.length == 0) return 0;
    // ascending sorting by end
    Arrays.sort(intvs, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) {
            return a[1] - b[1];
        }
    });
    // at least have one interval without intersection
    int count = 1;
    // after sorting, the first interval is x
    int x_end = intvs[0][1];
    for (int[] interval : intvs) {
        int start = interval[0];
        if (start >= x_end) {
            // get the next selected interval
            count++;
            x_end = interval[1];
        }
    }
    return count;
}
'''
## KMP
'''
Many readers complain that the KMP algorithm is incomprehensible. This is normal. When I think about the KMP algorithm explained in university textbooks, I don't know how many future Knuth, Morris, Pratt will be dismissed in advance. Some excellent students use the process of pushing the KMP algorithm to help understand the algorithm. This is a way, but this article will help the reader understand the principle of the algorithm from a logical level. Between ten lines of code, KMP died.

public class KMP {
    private int[][] dp;
    private String pat;

    public KMP(String pat) {
        this.pat = pat;
        int M = pat.length();
        // dp[state][character] = next state
        dp = new int[M][256];
        // base case
        dp[0][pat.charAt(0)] = 1;
        // Shadow state X is initially 0
        int X = 0;
        // Build state transition diagram (slightly more compact)
        for (int j = 1; j < M; j++) {
            for (int c = 0; c < 256; c++)
                dp[j][c] = dp[X][c];
            dp[j][pat.charAt(j)] = j + 1;
            // Update shadow status
            X = dp[X][pat.charAt(j)];
        }
    }

    public int search(String txt) {
        int M = pat.length();
        int N = txt.length();
        // The initial state of pat is 0
        int j = 0;
        for (int i = 0; i < N; i++) {
            // Calculate the next state of pat
            j = dp[j][txt.charAt(i)];
            // Reached termination state and returned results
            if (j == M) return i - M + 1;
        }
        // Not reached termination state, matching failed
        return -1;
    }
}
'''

## Stock Buy and Sell Problems
'''
This article uses the techniques of state machines to solve it.
The first problem is that only one transaction is performed, which is equivalent to k = 1. The second problem is that the number of transactions is unlimited, which is equivalent to k = +infinity (positive infinity). The third problem is that only two transactions are performed, which is equivalent to k = 2. The remaining two are also unlimited, but the additional conditions of the "freezing period" and "handling fee" for the transaction are actually variants of the second problem, which are easy to handle.
'''
# k = 1
'''
Instead of the entire DP array, only one variable is needed to store the adjacent state, which can reduce the space complexity to O(1):

// k == 1
int maxProfit_k_1(int[] prices) {
    int n = prices.length;
    // base case: dp[-1][0] = 0, dp[-1][1] = -infinity
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    for (int i = 0; i < n; i++) {
        // dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        // dp[i][1] = max(dp[i-1][1], -prices[i])
        dp_i_1 = Math.max(dp_i_1, -prices[i]);
    }
    return dp_i_0;
}
'''
# k = +infinity
'''
int maxProfit_k_inf(int[] prices) {
    int n = prices.length;
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    for (int i = 0; i < n; i++) {
        int temp = dp_i_0;
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = Math.max(dp_i_1, temp - prices[i]);
    }
    return dp_i_0;
}
'''
# k = +infinity with cooldown
'''
int maxProfit_with_cool(int[] prices) {
    int n = prices.length;
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    int dp_pre_0 = 0; // variable representing dp[i-2][0]
    for (int i = 0; i < n; i++) {
        int temp = dp_i_0;
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = Math.max(dp_i_1, dp_pre_0 - prices[i]);
        dp_pre_0 = temp;
    }
    return dp_i_0;
}
'''
# k = +infinity with fee
'''
int maxProfit_with_fee(int[] prices, int fee) {
    int n = prices.length;
    int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
    for (int i = 0; i < n; i++) {
        int temp = dp_i_0;
        dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
        dp_i_1 = Math.max(dp_i_1, temp - prices[i] - fee);
    }
    return dp_i_0;
}
'''
# k = 2
'''
int max_k = 2;
int[][][] dp = new int[n][max_k + 1][2];
for (int i = 0; i < n; i++) {
    for (int k = max_k; k >= 1; k--) {
        if (i - 1 == -1) { /* Deal with the base case */ }
        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
    }
}
// Exhaust n × max_k × 2 states, correct!
return dp[n - 1][max_k][0];
'''

# k = any integer
'''
int maxProfit_k_any(int max_k, int[] prices) {
    int n = prices.length;
    if (max_k > n / 2) 
        return maxProfit_k_inf(prices);

    int[][][] dp = new int[n][max_k + 1][2];
    for (int i = 0; i < n; i++) 
        for (int k = max_k; k >= 1; k--) {
            if (i - 1 == -1) { /* Deal with the base case */ }
            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
            dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);     
        }
    return dp[n - 1][max_k][0];
}
'''

## House Robber Problems
# Recursive
'''
// main function
public int rob(int[] nums) {
    return dp(nums, 0);
}
// return nums[start..] Maximum value that can be robbed
private int dp(int[] nums, int start) {
    if (start >= nums.length) {
        return 0;
    }

    int res = Math.max(
            // not rob, walk to the next house
            dp(nums, start + 1), 
            // rob，walk to the house after next
            nums[start] + dp(nums, start + 2)
        );
    return res;
}
'''
# Memo
'''
private int[] memo;
// main function
public int rob(int[] nums) {
    // initialize the memos
    memo = new int[nums.length];
    Arrays.fill(memo, -1);
    // robber robs from house 0
    return dp(nums, 0);
}

// return dp[start..] Maximum value that can be robbed
private int dp(int[] nums, int start) {
    if (start >= nums.length) {
        return 0;
    }
    // Avoid duplicate processing
    if (memo[start] != -1) return memo[start];

    int res = Math.max(dp(nums, start + 1), 
                    nums[start] + dp(nums, start + 2));
    // record the result to memos
    memo[start] = res;
    return res;
}
'''

# Bottom up
'''
 int rob(int[] nums) {
    int n = nums.length;
    // dp[i] = x: start rob at i-th house, the maximum money you can get is x
    // base case: dp[n] = 0
    int[] dp = new int[n + 2];
    for (int i = n - 1; i >= 0; i--) {
        dp[i] = Math.max(dp[i + 1], nums[i] + dp[i + 2]);
    }
    return dp[0];
}
'''

# Bottom up O(1)
'''
int rob(int[] nums) {
    int n = nums.length;
    // record dp[i+1] and dp[i+2]
    int dp_i_1 = 0, dp_i_2 = 0;
    // record dp[i]
    int dp_i = 0; 
    for (int i = n - 1; i >= 0; i--) {
        dp_i = Math.max(dp_i_1, nums[i] + dp_i_2);
        dp_i_2 = dp_i_1;
        dp_i_1 = dp_i;
    }
    return dp_i;
}
'''
## House Robber II
'''
The input is still an array, but these houses are not in a row but arranged in a circle.

First of all, the first and last rooms cannot be robbed at the same time, then there are only three possible situations: case I, either they are not robbed; case II the first house is robbed and the last one is not robbed; case III, the first house is not robbed and the last one is robbed.
he solution is the maximum of these three cases. However, in fact, we don't need to compare three cases, just compare case II and case III. Because these two cases have more room to choose than the case I, the money in the house is non-negative. So the optimal decision result is certainly not small if we have more choice.
'''
'''
public int rob(int[] nums) {
    int n = nums.length;
    if (n == 1) return nums[0];
    return Math.max(robRange(nums, 0, n - 2), 
                    robRange(nums, 1, n - 1));
}

// Calculate the optimal result for only the closed interval [start, end]
int robRange(int[] nums, int start, int end) {
    int n = nums.length;
    int dp_i_1 = 0, dp_i_2 = 0;
    int dp_i = 0;
    for (int i = end; i >= start; i--) {
        dp_i = Math.max(dp_i_1, nums[i] + dp_i_2);
        dp_i_2 = dp_i_1;
        dp_i_1 = dp_i;
    }
    return dp_i;
}
'''

## House Robber III
'''
The house now is arranged not a row, not a circle, but a binary tree! The house is on the node of the binary tree. The two connected houses cannot be robbed at the same time. 
'''
'''
Map<TreeNode, Integer> memo = new HashMap<>();
public int rob(TreeNode root) {
    if (root == null) return 0;
    // Eliminating overlapping subproblems with memos
    if (memo.containsKey(root)) 
        return memo.get(root);
    // rob, walk to the house after next
    int do_it = root.val
        + (root.left == null ? 
            0 : rob(root.left.left) + rob(root.left.right))
        + (root.right == null ? 
            0 : rob(root.right.left) + rob(root.right.right));
    // not rob, walk to the next house
    int not_do = rob(root.left) + rob(root.right);

    int res = Math.max(do_it, not_do);
    memo.put(root, res);
    return res;
}
'''
'''
int rob(TreeNode root) {
    int[] res = dp(root);
    return Math.max(res[0], res[1]);
}

/* return an array of size 2 arr
arr [0] means the maximum amount of money you get if you do not rob root
arr [1] means the maximum amount of money you get if you rob root */
int[] dp(TreeNode root) {
    if (root == null)
        return new int[]{0, 0};
    int[] left = dp(root.left);
    int[] right = dp(root.right);
    // rob, walk to the house after next
    int rob = root.val + left[0] + right[0];
    // not rob, The next home can be robbed or not, depending on the size of the income
    int not_rob = Math.max(left[0], left[1])
                + Math.max(right[0], right[1]);

    return new int[]{not_rob, rob};
}
'''

## 4 Keys Keyboard
'''
what choices are obvious for each keystroke: four types are the 4 keys mentioned in the title, which are A, Ctrl-A, Ctrl-C, and Ctrl-V.
Now you think about it, Is it correct for me to define the status of this problem as follows?
The first state is the remaining number of times the key can be pressed, we use n to represent it.
The second state are the number of characters 'A' on the current screen, we use a_num.
The third state are the number of characters 'A' still in the clipboard, represented by copy.
By defining state in this way, we can know the base case: when the number of remaining n is 0, a_num is the answer we want.
Combining the four choices just mentioned, we can express these kinds of choices of state transitions:
dp(n - 1, a_num + 1, copy)        # [ A ]
# comment: Press the 'A' key to add a character to the screen.
# Subtract 1 at the same time the number of times you are allowed to press the keyboard.

dp(n - 1, a_num + copy, copy)     # [Ctrl-V]
# comment: Press C-V to paste, the characters in the clipboard are added to the screen.
# Subtract 1 at the same time the number of times you are allowed to press the keyboard.

dp(n - 2, a_num, a_num)           # [Ctrl-A] & [Ctrl-C]
# comment: Ctrl + A and Ctrl + C can obviously be used together.
# The number of 'A' in the clipboard becomes the number of 'A' on the screen.
# Subtract 2 at the same time the number of times you are allowed to press the keyboard.
'''
# Recursive
def maxA(N: int) -> int:

    # It can be verified that for the initial (n, a_num, copy) state,
    # there can be at most dp (n, a_num, copy) 'A' on the screen.
    def dp(n, a_num, copy):
        # base case
        if n <= 0: return a_num;
        # Let ’s try all three options and choose the largest one.
        return max(
                dp(n - 1, a_num + 1, copy),    # [ A ]
                dp(n - 1, a_num + copy, copy), # [Ctrl-V]
                dp(n - 2, a_num, a_num)        # [Ctrl-A] & [Ctrl-C]
            )

    # You can press the key n times, then there is no 'A' in the screen
    # and the clipboard.
    return dp(N, 0, 0)

# Memo
def maxA(N: int) -> int:
    # memorandum
    memo = dict()
    def dp(n, a_num, copy):
        if n <= 0: return a_num;
        # Avoid overlapping subproblems being recalculated
        if (n, a_num, copy) in memo:
            return memo[(n, a_num, copy)]

        memo[(n, a_num, copy)] = max(
                # These options are still the same
            )
        return memo[(n, a_num, copy)]

    return dp(N, 0, 0)

# Bottom up
'''
public int maxA(int N) {
    int[] dp = new int[N + 1];
    dp[0] = 0;
    for (int i = 1; i <= N; i++) {
        // press [ A ]
        dp[i] = dp[i - 1] + 1;
        for (int j = 2; j < i; j++) {
            // [Ctrl-A] & [Ctrl-C] -> dp[j-2], Paste i-j times
            // There are { dp[j-2] * (i-j+1) }number of 'A' on the screen
            dp[i] = Math.max(dp[i], dp[j - 2] * (i - j + 1));
        }
    }
    // What is the maximum number of 'A' after N keystrokes?
    return dp[N];
}
'''
## Regular Expression
'''
Given a string(s) and a string mode(p). Implement the Regular Expression that supports the '.' and '*' match.
'.'  matches any single character
'*'  matches zero or more before "*"
Note: Matches should cover the whole string, not part of it.
'''
# Recursion with memo
def isMatch(text, pattern) -> bool:
    memo = dict() # memo
    def dp(i, j):
        if (i, j) in memo: return memo[(i, j)]
        if j == len(pattern): return i == len(text)

        first = i < len(text) and pattern[j] in {text[i], '.'}

        if j <= len(pattern) - 2 and pattern[j + 1] == '*':
            ans = dp(i, j + 2) or \
                    first and dp(i + 1, j)
        else:
            ans = first and dp(i + 1, j + 1)

        memo[(i, j)] = ans
        return ans

    return dp(0, 0)

# brute force recursive
def isMatch(text, pattern) -> bool:
    if not pattern: return not text

    first = bool(text) and pattern[0] in {text[0], '.'}

    if len(pattern) >= 2 and pattern[1] == '*':
        return isMatch(text, pattern[2:]) or \
                first and isMatch(text[1:], pattern)
    else:
        return first and isMatch(text[1:], pattern[1:])
    
## Longest Increasing Subsequence
'''
Note the difference between the two terms "subsequence" and "substring". Substrings must be continuous, and subsequences are not necessarily continuous. 
'''
# 1
'''
DP n^2

public int lengthOfLIS(int[] nums) {
    int[] dp = new int[nums.length];
    // set all the dp values to 1
    Arrays.fill(dp, 1);
    for (int i = 0; i < nums.length; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) 
                dp[i] = Math.max(dp[i], dp[j] + 1);
        }
    }

    int res = 0;
    for (int i = 0; i < dp.length; i++) {
        res = Math.max(res, dp[i]);
    }
    return res;
}
'''

# Binary Search
'''
public int lengthOfLIS(int[] nums) {
    int[] top = new int[nums.length];
    // Initialize the number of piles
    int piles = 0;
    for (int i = 0; i < nums.length; i++) {
        // play cards to be handled
        int poker = nums[i];

        /***** binary search *****/
        int left = 0, right = piles;
        while (left < right) {
            int mid = (left + right) / 2;
            if (top[mid] > poker) {
                right = mid;
            } else if (top[mid] < poker) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        /*********************************/

        // create a new pile and put this card into it
        if (left == piles) piles++;
        // put the card on the top
        top[left] = poker;
    }
    // The number of piles represents the length of LIS
    return piles;
}
'''

### *** Backtracking
## Framework
'''
Solving a backtracking problem is actually a traversal process of a decision tree. Now you only need to think about 3 terms:
- Path: the selection that have been made.
- Selection List: the selection you can currently make.
- End Condition: the condition under which you reach the bottom of the decision tree, and can no longer make a selection.
Pseudocode:
result = []
def backtrack(Path, Seletion List  ):
    if meet the End Conditon:
        result.add(Path)
        return

    for seletion in Seletion List:
        select
        backtrack(Path, Seletion List)
        deselect

The core is the recursion in the for loop. It makes a selection before the recursive call and undoes the selection after the recursive call.
'''

## find out all the permutations
'''
List<List<Integer>> res = new LinkedList<>();

/* The main method, enter a set of unique numbers and return their full permutations */
List<List<Integer>> permute(int[] nums) {
    // record Path
    LinkedList<Integer> track = new LinkedList<>();
    backtrack(nums, track);
    return res;
}

// Path: recorded in track
// Seletion List: those elements in nums that do not exist in track
// End Condition: all elements in nums appear in track
void backtrack(int[] nums, LinkedList<Integer> track) {
    // trigger the End Condition
    if (track.size() == nums.length) {
        res.add(new LinkedList(track));
        return;
    }

    for (int i = 0; i < nums.length; i++) {
        // exclude illegal seletions
        if (track.contains(nums[i]))
            continue;
        // select
        track.add(nums[i]);
        // enter the next level decision tree
        backtrack(nums, track);
        // deselect
        track.removeLast();
    }
}
'''

## N Queen
'''
vector<vector<string>> res;

/* Enter board length n, return all legal placements */
vector<vector<string>> solveNQueens(int n) {
    // '.' Means empty, and 'Q' means queen, initializing the empty board.
    vector<string> board(n, string(n, '.'));
    backtrack(board, 0);
    return res;
}

// Path:The rows smaller than row in the board have been successfully placed the queens
// Seletion List: all columns in 'rowth' row are queen's seletions
// End condition: row meets the last line of board(n)
void backtrack(vector<string>& board, int row) {
    // trigger the End Condition
    if (row == board.size()) {
        res.push_back(board);
        return;
    }

    int n = board[row].size();
    for (int col = 0; col < n; col++) {
        // exclude illegal seletions
        if (!isValid(board, row, col)) 
            continue;
        // select
        board[row][col] = 'Q';
        // enter next row decision
        backtrack(board, row + 1);
        // deselect
        board[row][col] = '.';
    }
}

/*Is it possible to place a queen on board [row] [col]? */
bool isValid(vector<string>& board, int row, int col) {
    int n = board.size();
    // Check if share the same column
    for (int i = 0; i < n; i++) {
        if (board[i][col] == 'Q')
            return false;
    }
    // Check if share the same right diagonal
    for (int i = row - 1, j = col + 1; 
            i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q')
            return false;
    }
    // Check if share the same left diagonal
    for (int i = row - 1, j = col - 1;
            i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q')
            return false;
    }
    return true;
}

When we don't want to get all legal answers but only one answer, what should we do ?

// Returns true after finding an answer
bool backtrack(vector<string>& board, int row) {
    // Trigger End Condition
    if (row == board.size()) {
        res.push_back(board);
        return true;
    }
    ...
    for (int col = 0; col < n; col++) {
        ...
        board[row][col] = 'Q';

        if (backtrack(board, row + 1))
            return true;

        board[row][col] = '.';
    }

    return false;
}
'''

### *** Binary Search
## Framework
'''
int binarySearch(int[] nums, int target) {
    int left = 0, right = ...;

    while(...) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            ...
        } else if (nums[mid] < target) {
            left = ...
        } else if (nums[mid] > target) {
            right = ...
        }
    }
    return ...;
}
'''

## basic binary search
'''
int binarySearch(int[] nums, int target) {
    int left = 0; 
    int right = nums.length - 1; // attention

    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(nums[mid] == target)
            return mid; 
        else if (nums[mid] < target)
            left = mid + 1; // attention
        else if (nums[mid] > target)
            right = mid - 1; // attention
    }
    return -1;
}
'''

## binary search to find the left border
'''
int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    // search interval is [left, right]
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            // search interval is [mid+1, right]
            left = mid + 1;
        } else if (nums[mid] > target) {
            // search interval is [left, mid-1]
            right = mid - 1;
        } else if (nums[mid] == target) {
            // shrink right border
            right = mid - 1;
        }
    }
    // check out of bounds
    if (left >= nums.length || nums[left] != target)
        return -1;
    return left;
}
'''

## binary search to find the right border
'''
int right_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // here~ change to shrink left bounds
            left = mid + 1;
        }
    }
    // here~ change to check right out of bounds, see below 
    if (right < 0 || nums[right] != target)
        return -1;
    return right;
}
'''

## Unified 
'''
int binary_search(int[] nums, int target) {
    int left = 0, right = nums.length - 1; 
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1; 
        } else if(nums[mid] == target) {
            // Return directly
            return mid;
        }
    }
    // Return directly
    return -1;
}

int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // Don't return! Lock left border
            right = mid - 1;
        }
    }
    // Check whether left border out of bounds lastly
    if (left >= nums.length || nums[left] != target)
        return -1;
    return left;
}


int right_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] == target) {
            // Don't return! Lock right border
            left = mid + 1;
        }
    }
    // Check whether right border out of bounds lastly
    if (right < 0 || nums[right] != target)
        return -1;
    return right;
}
'''

