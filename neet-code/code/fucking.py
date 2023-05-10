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

### *** Data Structures
## Heap
class Heap:
    def __init__(self, comparator):
        self.arr = []
        self.comparator = comparator

    def get_parent(self, k):
        return (k - 1) // 2

    def peek(self):
        return self.arr[0]

    def push(self, v):
        self.arr.append(v)
        self._heapify_up(len(self.arr) - 1)

    def pop(self):
        self.arr[0] = self.arr[-1]
        self.arr.pop()
        self._heapify_down(0)

    def _heapify_up(self, k):
        parent = self.get_parent(k)
        if parent == -1:
            return
        if not self.comparator(self.arr[parent], self.arr[k]):
            self.arr[parent], self.arr[k] = self.arr[k], self.arr[parent]
            self._heapify_up(parent)

    def _heapify_down(self, k):
        if k >= len(self.arr):
            return
        left = 2 * k + 1
        right = 2 * k + 2
        index = k
        if left < len(self.arr) and not self.comparator(self.arr[index], self.arr[left]):
            index = left
        if right < len(self.arr) and not self.comparator(self.arr[index], self.arr[right]):
            index = right
        if index != k:
            self.arr[index], self.arr[k] = self.arr[k], self.arr[index]
            self._heapify_down(index)

    def make_heap(self, arr):
        for _, v in enumerate(arr):
            self.arr.append(v)

        index = (len(self.arr) - 1) // 2
        for i in range(index, -1, -1):
            self._heapify_down(i)

min_heap = Heap(lambda a, b : a < b)
max_heap = Heap(lambda a, b : a > b)

arr = [4, 3, 5, 2, 1, 6, 8, 7, 9]
min_heap.make_heap(arr)
max_heap.make_heap(arr)

print(min_heap.peek())
print(max_heap.peek())

min_heap.push(0)
max_heap.push(10)

print(min_heap.peek())
print(max_heap.peek())

min_heap.pop()
max_heap.pop()

print(min_heap.peek())
print(max_heap.peek())

## LRU
'''
class Node {
    public int key, val;
    public Node next, prev;
    public Node(int k, int v) {
        this.key = k;
        this.val = v;
    }
}

class DoubleList {  
    // Add x at the head, time complexity O(1)
    public void addFirst(Node x);

    // Delete node x in the linked list (x is guaranteed to exist)
    // Given a node in a doubly linked list, time complexity O(1)
    public void remove(Node x);

    // Delete and return the last node in the linked list, time complexity O(1)
    public Node removeLast();

    // Return the length of the linked list, time complexity O(1)
    public int size();
}

class LRUCache {
    // key -> Node(key, val)
    private HashMap<Integer, Node> map;
    // Node(k1, v1) <-> Node(k2, v2)...
    private DoubleList cache;
    // Max capacity
    private int cap;

    public LRUCache(int capacity) {
        this.cap = capacity;
        map = new HashMap<>();
        cache = new DoubleList();
    }

    public int get(int key) {
        if (!map.containsKey(key))
            return -1;
        int val = map.get(key).val;
        // Using put method to bring it forward to the head
        put(key, val);
        return val;
    }

    public void put(int key, int val) {
        // Initialize new node x
        Node x = new Node(key, val);

        if (map.containsKey(key)) {
            // Delete the old node, add to the head
            cache.remove(map.get(key));
            cache.addFirst(x);
            // Update the corresponding record in map
            map.put(key, x);
        } else {
            if (cap == cache.size()) {
                // Delete the last node in the linked list
                Node last = cache.removeLast();
                map.remove(last.key);
            }
            // Add to the head
            cache.addFirst(x);
            map.put(key, x);
        }
    }
}
'''

## Binary Search Operations
'''
# Traverse
void BST(TreeNode root, int target) {
    if (root.val == target)
        // When you find the target, your manipulation should be written here
    if (root.val < target) 
        BST(root.right, target);
    if (root.val > target)
        BST(root.left, target);
}
# Same Tree
boolean isSameTree(TreeNode root1, TreeNode root2) {
    // If they are null, they are identical obviously
    if (root1 == null && root2 == null) return true;
    // If one of the nodes is void, but the other is not null, they are not identical
    if (root1 == null || root2 == null) return false;
    // If they are all not void, but their values are not equal, they are not identical
    if (root1.val != root2.val) return false;

    // To recursively compare every pair of the node
    return isSameTree(root1.left, root2.left)
        && isSameTree(root1.right, root2.right);
}

# Valid BST
boolean isValidBST(TreeNode root) {
    return isValidBST(root, null, null);
}

boolean isValidBST(TreeNode root, TreeNode min, TreeNode max) {
    if (root == null) return true;
    if (min != null && root.val <= min.val) return false;
    if (max != null && root.val >= max.val) return false;
    return isValidBST(root.left, min, root) 
        && isValidBST(root.right, root, max);
}

# In BST
boolean isInBST(TreeNode root, int target) {
    if (root == null) return false;
    if (root.val == target)
        return true;
    if (root.val < target) 
        return isInBST(root.right, target);
    if (root.val > target)
        return isInBST(root.left, target);
    // The manipulations in the root node are finished, and the framework is done, great!
}

# Delete
TreeNode deleteNode(TreeNode root, int key) {
    if (root == null) return null;
    if (root.val == key) {
        // These two IF function handle the situation 1 and situation 2
        if (root.left == null) return root.right;
        if (root.right == null) return root.left;
        // Deal with situation 3
        TreeNode minNode = getMin(root.right);
        root.val = minNode.val;
        root.right = deleteNode(root.right, minNode.val);
    } else if (root.val > key) {
        root.left = deleteNode(root.left, key);
    } else if (root.val < key) {
        root.right = deleteNode(root.right, key);
    }
    return root;
}

TreeNode getMin(TreeNode node) {
    // The left child node is the minimum
    while (node.left != null) node = node.left;
    return node;
}
'''

## Monotonic Stack
'''
Monotonic stack is actually a stack. It just uses some ingenious logic to keep the elements in the stack orderly (monotone increasing or monotone decreasing) after each new element putting into the stack.
Well,sounds like a heap? No, monotonic stack is not widely used. It only deals with one typical problem, which is called Next Greater Element.
Template for monotonic queue solving problem. The for loop scans elements from the back to the front,and while we use the stack structure and enter the stack back to front, we are actually going to exit the stack front to back. The while loop is to rule out the elements between the two "tall" elements.Their existence has no meaning, because there is a "taller" element in front of them and they cannot be considered as the Next Great Number of the subsequent elements.
The time complexity of this algorithm is not so intuitive. If you see for loop nesting with while loop, you may think that the complexity of this algorithm is O(n^2), but in fact the complexity of this algorithm is only O(n).
vector<int> nextGreaterElement(vector<int>& nums) {
    vector<int> ans(nums.size()); // array to store answer
    stack<int> s;
    for (int i = nums.size() - 1; i >= 0; i--) { // put it into the stack back to front
        while (!s.empty() && s.top() <= nums[i]) { // determine by height
            s.pop(); // short one go away while blocked
        }
        ans[i] = s.empty() ? -1 : s.top(); // the first tall behind this element
        s.push(nums[i]); // get into the queue and wait for later height determination
    }
    return ans;
}

## 
Give you an array T = [73, 74, 75, 71, 69, 72, 76, 73], which stores the weather temperature in recent days(Is it in teppanyaki? No, it's in Fahrenheit). You return an array to calculate: for each day, how many days do you have to wait for a warmer temperature;and if you can't wait for that day, fill in 0.
vector<int> dailyTemperatures(vector<int>& T) {
    vector<int> ans(T.size());
    stack<int> s; // here for element index，not element
    for (int i = T.size() - 1; i >= 0; i--) {
        while (!s.empty() && T[s.top()] <= T[i]) {
            s.pop();
        }
        ans[i] = s.empty() ? 0 : (s.top() - i); // get index spacing
        s.push(i); // add index，not element
    }
    return ans;
}

## Circular Array
After adding the ring attribute, the difficulty lies in that the meaning of Next is not only the right side of the current element, but also the left side of the current element (as shown in the above example).

vector<int> nextGreaterElements(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n); // store result
    stack<int> s;
    // pretend that this array is doubled in length. Instead of constructing a new array, we can use the technique of circular array to simulate.
    for (int i = 2 * n - 1; i >= 0; i--) {
        while (!s.empty() && s.top() <= nums[i % n])
            s.pop();
        res[i % n] = s.empty() ? -1 : s.top();
        s.push(nums[i % n]);
    }
    return res;
}
'''

## Monotonic Queue
'''
The core idea of "monotonic queue" is similar to "monotonic stack". The push method of the monotonic queue still adds elements to the end of the queue, but deletes the previous elements smaller than the new element.

class MonotonicQueue {
    // add element n to the end of the line
    void push(int n);
    // returns the maximum value in the current queue
    int max();
    // if the head element is n, delete it
    void pop(int n);
}

class MonotonicQueue {
private:
    deque<int> data;
public:
    void push(int n) {
        while (!data.empty() && data.back() < n) 
            data.pop_back();
        data.push_back(n);
    }

    int max() { return data.front(); }

    void pop(int n) {
        if (!data.empty() && data.front() == n)
            data.pop_front();
    }
};

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    MonotonicQueue window;
    vector<int> res;
    for (int i = 0; i < nums.size(); i++) {
        if (i < k - 1) { // fill the first k-1 of the window first
            window.push(nums[i]);
        } else { // window slide forward
            window.push(nums[i]);
            res.push_back(window.max());
            window.pop(nums[i - k + 1]);
        }
    }
    return res;
}
'''

## Design Twitter
'''
class Twitter {

    /** user post a tweet */
    public void postTweet(int userId, int tweetId) {}

    /** return the list of IDs of recent tweets, 
    from the users that the current user follows (including him/herself),
    maximum 10 tweets with updated time sorted in descending order */
    public List<Integer> getNewsFeed(int userId) {}

    /** follower will follow the followee,
    create the ID if it doesn't exist */
    public void follow(int followerId, int followeeId) {}

    /** follower unfollows the followee,
    do nothing if the ID does not exist */
    public void unfollow(int followerId, int followeeId) {}
}


// static int timestamp = 0
class User {
    private int id;
    public Set<Integer> followed;
    // The head of the linked list of posted tweets by the user
    public Tweet head;

    public User(int userId) {
        followed = new HashSet<>();
        this.id = userId;
        this.head = null;
        // follow the user him/herself
        follow(id);
    }

    public void follow(int userId) {
        followed.add(userId);
    }

    public void unfollow(int userId) {
        // a user is not allowed to unfollow him/herself
        if (userId != this.id)
            followed.remove(userId);
    }

    public void post(int tweetId) {
        Tweet twt = new Tweet(tweetId, timestamp);
        timestamp++;
        // insert the new tweet to the head of the linked list
        // the closer a tweet is to the head, the larger the value of time
        twt.next = head;
        head = twt;
    }
}

class Tweet {
    private int id;
    private int time;
    private Tweet next;

    // initialize with tweet ID and post timestamp
    public Tweet(int id, int time) {
        this.id = id;
        this.time = time;
        this.next = null;
    }
}

class Twitter {
    private static int timestamp = 0;
    private static class Tweet {...}
    private static class User {...}

    // we need a mapping to associate userId and User
    private HashMap<Integer, User> userMap = new HashMap<>();

    /** user posts a tweet */
    public void postTweet(int userId, int tweetId) {
        // instantiate an instance if userId does not exist
        if (!userMap.containsKey(userId))
            userMap.put(userId, new User(userId));
        User u = userMap.get(userId);
        u.post(tweetId);
    }

    /** follower follows the followee */
    public void follow(int followerId, int followeeId) {
        // instantiate if the follower does not exist
        if(!userMap.containsKey(followerId)){
            User u = new User(followerId);
            userMap.put(followerId, u);
        }
        // instantiate if the followee does not exist
        if(!userMap.containsKey(followeeId)){
            User u = new User(followeeId);
            userMap.put(followeeId, u);
        }
        userMap.get(followerId).follow(followeeId);
    }

    /** follower unfollows the followee, do nothing if follower does not exists */
    public void unfollow(int followerId, int followeeId) {
        if (userMap.containsKey(followerId)) {
            User flwer = userMap.get(followerId);
            flwer.unfollow(followeeId);
        }
    }

    /** return the list of IDs of recent tweets, 
    from the users that the current user follows (including him/herself),
    maximum 10 tweets with updated time sorted in descending order */
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        if (!userMap.containsKey(userId)) return res;
        // IDs of followees
        Set<Integer> users = userMap.get(userId).followed;
        // auto sorted by time property in descending order
        // the size will be equivalent to users
        PriorityQueue<Tweet> pq = 
            new PriorityQueue<>(users.size(), (a, b)->(b.time - a.time));

        // first, insert all heads of linked list into the priority queue
        for (int id : users) {
            Tweet twt = userMap.get(id).head;
            if (twt == null) continue;
            pq.add(twt);
        }

        while (!pq.isEmpty()) {
            // return only 10 records
            if (res.size() == 10) break;
            // pop the tweet with the largest time (the most recent)
            Tweet twt = pq.poll();
            res.add(twt.id);
            // insert the next tweet, which will be sorted automatically
            if (twt.next != null) 
                pq.add(twt.next);
        }
        return res;
    }
}
'''

## Reverse Part of Linked List via Recursion
'''
The idea is similar to reversing the whole linked list, only a few modifications needed:
Main differences:
Base case n == 1, if reverse only one element, then new head is itself, meanwhile remember to mark the successor node.
In previouse solution, we set head.next directly to null, because after reversing the whole list, head becoms the last node. But now head may not be the last node after reversion, so we need mark successor (the (n+1)th node), and link it to head after reversion.

ListNode successor = null; // successor node

// reverse n nodes starting from head, and return new head
ListNode reverseN(ListNode head, int n) {
    if (n == 1) { 
        // mark the (n + 1)th node
        successor = head.next;
        return head;
    }
    // starts from head.next, revers the first n - 1 nodes
    ListNode last = reverseN(head.next, n - 1);

    head.next.next = head;
    // link the new head to successor
    head.next = successor;
    return last;
}

ListNode reverseBetween(ListNode head, int m, int n) {
    // base case
    if (m == 1) {
        return reverseN(head, n);
    }
    head.next = reverseBetween(head.next, m - 1, n - 1);
    return head;
}
'''

## Queue Implement Stack/Stack implement Queue
'''
https://labuladong.gitbook.io/algo-en/ii.-data-structure/implementqueueusingstacksimplementstackusingqueues
'''

### *** Algo Thinking
## Framework for Backtracking Algorithm
'''
Solving a backtracking problem is actually a traversal process of a decision tree. Now you only need to think about 3 terms:
- Path: the selection that have been made.
- Selection List: the selection you can currently make.
- End Condition: the condition under which you reach the bottom of the decision tree, and can no longer make a selection.
The core is the recursion in the for loop. It makes a selection before the recursive call and undoes the selection after the recursive call.
Pseudo code:
result = []
def backtrack(Path, Seletion List  ):
    if meet the End Conditon:
        result.add(Path)
        return

    for seletion in Seletion List:
        select
        backtrack(Path, Seletion List)
        deselect

# full permutation
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

# N Queen Problem
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
'''

## Binary Search
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

## The Tech of Double Pointer
'''
I divided the double pointer technique into two categories, one is "fast and slow pointer" and the other is "left and right pointer". The former solution mainly solves the problems in the linked list, such as determining whether the linked list contains a ring; the latter mainly solves the problems in the array (or string), such as binary search.
# First, the common algorithm of fast and slow pointers
# Determine whether the linked list contains a ring
boolean hasCycle(ListNode head) {
    ListNode fast, slow;
    fast = slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;

        if (fast == slow) return true;
    }
    return false;
}

# Knowing that the linked list contains a ring, return to the starting position of the ring
ListNode detectCycle(ListNode head) {
    ListNode fast, slow;
    fast = slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
        if (fast == slow) break;
    }
    // The above code is similar to the hasCycle function
    slow = head;
    while (slow != fast) {
        fast = fast.next;
        slow = slow.next;
    }
    return slow;
}

# Find the midpoint of the linked list
while (fast != null && fast.next != null) {
    fast = fast.next.next;
    slow = slow.next;
}
// "slow" is in the middle
return slow;

# Find the k-th element from the bottom of the linked list
ListNode slow, fast;
slow = fast = head;
while (k-- > 0) 
    fast = fast.next;

while (fast != null) {
    slow = slow.next;
    fast = fast.next;
}
return slow;

# -- Second, the common algorithm of left and right pointer
# Binary Search
int binarySearch(int[] nums, int target) {
    int left = 0; 
    int right = nums.length - 1;
    while(left <= right) {
        int mid = (right + left) / 2;
        if(nums[mid] == target)
            return mid; 
        else if (nums[mid] < target)
            left = mid + 1; 
        else if (nums[mid] > target)
            right = mid - 1;
    }
    return -1;
}

# Two Sum
int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            //The index required for the question starts at 1
            return new int[]{left + 1, right + 1};
        } else if (sum < target) {
            left++; //Make "sum" bigger
        } else if (sum > target) {
            right--; // Make "sum" smaller
        }
    }
    return new int[]{-1, -1};
}

# Reverse the array
void reverse(int[] nums) {
    int left = 0;
    int right = nums.length - 1;
    while (left < right) {
        // swap(nums[left], nums[right])
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
        left++; right--;
    }
}

# Sliding window algorithm
This may be the highest state of the double pointer technique
'''

## Key Concept of TwoSum Problems
'''
I think the objective of the two sum problems is to tell us how to use the hash table.
# TwoSum I
Given an array of integers nums, and a specific integer target. Return indices of the two numbers such that they add up to target.
int[] twoSum(int[] nums, int target) {
    int n = nums.length;
    index<Integer, Integer> index = new HashMap<>();
    // Constructing a hash table: Elements are mapped to their corresponding indices
    for (int i = 0; i < n; i++)
        index.put(nums[i], i);

    for (int i = 0; i < n; i++) {
        int other = target - nums[i];
        // IF 'other' exists and it is not nums[i].
        if (index.containsKey(other) && index.get(other) != i)
            return new int[] {i, index.get(other)};
    }

    return new int[] {-1, -1};
}

# TwoSum II
design a class with two functions:
class TwoSum {
    // Add a 'number' to data structure
    public void add(int number);
    // Find out whether there exist two numbers and their sum is equal to 'value'.
    public boolean find(int value);
}

class TwoSum {
    Set<Integer> sum = new HashSet<>();
    List<Integer> nums = new ArrayList<>();

    public void add(int number) {
        // Recording all possible sum of two numbers
        for (int n : nums)
            sum.add(n + number);
        nums.add(number);
    }

    public boolean find(int value) {
        return sum.contains(value);
    }
}

# if the given array in TwoSum I is ordered, how do we design the algorithm? 
int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            return new int[]{left, right};
        } else if (sum < target) {
            left++; // Make sum bigger
        } else if (sum > target) {
            right--; // Make sum smaller
        }
    }
    // If no such two numbers exists
    return new int[]{-1, -1};
}
'''

## Implement a Calculator
'''
def calculate(s: str) -> int:

    def helper(s: List) -> int:
        stack = []
        sign = '+'
        num = 0

        while len(s) > 0:
            c = s.pop(0)
            if c.isdigit():
                num = 10 * num + int(c)
            # Meet the left parenthesis and start recursive calculation of num
            if c == '(':
                num = helper(s)

            if (not c.isdigit() and c != ' ') or len(s) == 0:
                if sign == '+': ...
                elif sign == '-': ... 
                elif sign == '*': ...
                elif sign == '/': ...
                num = 0
                sign = c
            # Return recursive result when encountering right parenthesis
            if c == ')': break
        return sum(stack)

    return helper(list(s))
'''

## Prefix Sum Skill
'''
Find the number of subarrays which sums to k.
---
The idea of Prefix Sum goes like this: for a given array nums, create another array to store the sum of prefix for pre-processing:
int n = nums.length;
// array of prefix sum
int[] preSum = new int[n + 1];
preSum[0] = 0;
for (int i = 0; i < n; i++)
    preSum[i + 1] = preSum[i] + nums[i];

The meaning of preSum is easy to understand. preSum[i] is the sum of nums[0..i-1]. If we want to calculate the sum of nums[i..j], we just need to perform preSum[j+1] - preSum[i] instead of traversing the whole subarray.
--
The idea of optimization is, to record down how many sum[j] equal to sum[i] - k such that we can update the result directly instead of having inner loop. We can utilize hash table to record both prefix sums and the frequency of each prefix sum.
--
int subarraySum(int[] nums, int k) {
    int n = nums.length;
    // map：prefix sum -> frequency
    HashMap<Integer, Integer> 
        preSum = new HashMap<>();
    // base case
    preSum.put(0, 1);

    int ans = 0, sum0_i = 0;
    for (int i = 0; i < n; i++) {
        sum0_i += nums[i];
        // this is the prefix sum we want to find nums[0..j]
        int sum0_j = sum0_i - k;
        // if it exists, we'll just update the result
        if (preSum.containsKey(sum0_j))
            ans += preSum.get(sum0_j);
        // record the prefix sum nums[0..i] and its frequency
        preSum.put(sum0_i, 
            preSum.getOrDefault(sum0_i, 0) + 1);
    }
    return ans;
}
'''

## FloodFill Algorithm
'''
# Framework
An array can be further abstracted as a graph. Hence, the problem becomes about traversing a graph, similar to traversing an N-ary tree. A few lines of code are enough to resolve the problem. Here is the framework:
// (x, y) represents the coordinate
void fill(int x, int y) {
    fill(x - 1, y); // up
    fill(x + 1, y); // down
    fill(x, y - 1); // left
    fill(x, y + 1); // right
}
Using this framework, we can resolve all problems about traversing a 2D array. The concept is also called Depth First Search (DFS), or quaternary (4-ary) tree traversal. The root node is coordinate (x, y). Its four child nodes are at root's four directions.
---
# Pay Attention to Details
Why is there infinite loop? Each coordinate needs to go through its 4 neighbors. Consequently, each coordinate will also be traversed 4 times by its 4 neighbors. When we visit an visited coordinate, we must guarantee to identify the situation and exit. If not, we'll go into infinite loop.
How to avoid the case of infinite loop? The most intuitive answer is to use a boolean 2D array of the same size as image, to record whether a coordinate has been traversed or not. If visited, return immediately.
---
int[][] floodFill(int[][] image,
        int sr, int sc, int newColor) {

    int origColor = image[sr][sc];
    fill(image, sr, sc, origColor, newColor);
    return image;
}

void fill(int[][] image, int x, int y,
        int origColor, int newColor) {
    // OUT: out of index
    if (!inArea(image, x, y)) return;
    // CLASH: meet other colors, beyond the area of origColor
    if (image[x][y] != origColor) return;
    // VISITED: visited origColor
    if (image[x][y] == -1) return;

    // choose: mark a flag as visited
    image[x][y] = -1;
    fill(image, x, y + 1, origColor, newColor);
    fill(image, x, y - 1, origColor, newColor);
    fill(image, x - 1, y, origColor, newColor);
    fill(image, x + 1, y, origColor, newColor);
    // unchoose: replace the mark with newColor
    image[x][y] = newColor;
}

boolean inArea(int[][] image, int x, int y) {
    return x >= 0 && x < image.length
        && y >= 0 && y < image[0].length;
}
'''

## Interval Scheduling: Interval Merging
'''
In the "Interval Scheduling: Greedy Algorithm", we use greedy algorithm to solve the interval scheduling problem, which means, given a lot of intervals, finding out the maximum subset without any overlapping.
---
The general thought for solving interval problems is observing regular patterns after the sorting process.

# intervals like [[1,3],[2,6]...]
def merge(intervals):
    if not intervals: return []
    # ascending sorting by start
    intervals.sort(key=lambda intv: intv[0])
    res = []
    res.append(intervals[0])

    for i in range(1, len(intervals)):
        curr = intervals[i]
        # quote of the last element in res
        last = res[-1]
        if curr[0] <= last[1]:
            # find the biggest end
            last[1] = max(last[1], curr[1])
        else:
            # address next interval need to be merged
            res.append(curr)
    return res
'''

## Interval Scheduling: Intersections of Intervals
'''
How to find out interval intersection from two set of intervals efficiently.
---
The general thought for interval problems is sorting first. Since question states that it has been ordered, then we can use two pointers to find out the intersections.

# A, B like [[0,2],[5,10]...]
def intervalIntersection(A, B):
    i, j = 0, 0 # double pointers
    res = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i][0], A[i][1]
        b1, b2 = B[j][0], B[j][1]
        # two intervals have intersection
        if b2 >= a1 and a2 >= b1:
            # compute the intersection and add it into res
            res.append([max(a1, b1), min(a2, b2)])
        # Pointer go forward
        if b2 < a2: j += 1
        else:       i += 1
    return res
'''

## String Multiplication
'''
For relatively small numbers, you can calculate directly using the operators provided by a programming language. When the numbers become very big, the default data types might overflow. An alternative way is to use string to represent the numbers, perform the multiplication in the primary school way, and produce the result as string as well. 

https://labuladong.gitbook.io/algo-en/iii.-algorithmic-thinking/string_multiplication
---
string multiply(string num1, string num2) {
    int m = num1.size(), n = num2.size();
    // the max number of digits in result is m + n
    vector<int> res(m + n, 0);
    // multiply from the rightmost digit
    for (int i = m - 1; i >= 0; i--)
        for (int j = n - 1; j >= 0; j--) {
            int mul = (num1[i]-'0') * (num2[j]-'0');
            // the corresponding index of product in res
            int p1 = i + j, p2 = i + j + 1;
            // add to res
            int sum = mul + res[p2];
            res[p2] = sum % 10;
            res[p1] += sum / 10;
        }
    // the result may have prefix of 0 (which is unused)
    int i = 0;
    while (i < res.size() && res[i] == 0)
        i++;
    // transform the result into string
    string str;
    for (; i < res.size(); i++)
        str.push_back('0' + res[i]);

    return str.size() == 0 ? "0" : str;
}
'''

## Pancake Sort
'''
To summarize, the idea is:
- Find the largest of the n pancakes.
- Move this largest pancake to the bottom.
- Recursively call pancakeSort(A, n-1).
Base case: When n == 1, there is no need to flip when sorting 1 pancake.
So, the last question left, how do you manage to turn a piece of pancake to the end?
In fact, it is very simple. For example, the third pancake is the largest, and we want to change it to the end, that is, to the n block. You can do this:
- Use a spatula to turn the first 3 pieces of pancakes, so that the largest pancake turns to the top.
- Use a spatula to flip all the first n cakes, so that the largest pancake turns to the n-th pancake, which is the last pancake.
---
// record the reverse operation sequence
LinkedList<Integer> res = new LinkedList<>();

List<Integer> pancakeSort(int[] cakes) {
    sort(cakes, cakes.length);
    return res;
}

void sort(int[] cakes, int n) {
    // base case
    if (n == 1) return;

    // find the index of the largest pancake
    int maxCake = 0;
    int maxCakeIndex = 0;
    for (int i = 0; i < n; i++)
        if (cakes[i] > maxCake) {
            maxCakeIndex = i;
            maxCake = cakes[i];
        }

    // first flip, turn the largest pancake to the top
    reverse(cakes, 0, maxCakeIndex);
    res.add(maxCakeIndex + 1);
    // second flip, turn the largest pancake to the bottom
    reverse(cakes, 0, n - 1);
    res.add(n);

    // recursive
    sort(cakes, n - 1);
}

void reverse(int[] arr, int i, int j) {
    while (i < j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        i++; j--;
    }
}
'''

## Sliding Window Algorithm
'''
The sliding window algorithm idea is like this:
We start with two pointers, left and right initially pointing to the first element of the string S.
We use the right pointer to expand the window [left, right] until we get a desirable window that contains all of the characters of T.
Once we have a window with all the characters, we can move the left pointer ahead one by one. If the window is still a desirable one we keep on updating the minimum window size.
If the window is not desirable any more, we repeat step 2 onwards.
This idea actually not difficult. Move right pointer to find a valid window. When a valid window is found, move left pointer to find a smaller window (optimal solution).
---
int left = 0, right = 0;

while (right < s.size()) {
    window.add(s[right]);
    right++;

    while (valid) {
        window.remove(s[left]);
        left++;
    }
}
---
# Minimum Window Substring
string minWindow(string s, string t) {
    // Records the starting position and length of the shortest substring
    int start = 0, minLen = INT_MAX;
    int left = 0, right = 0;

    unordered_map<char, int> window;
    unordered_map<char, int> needs;
    for (char c : t) needs[c]++;

    int match = 0;

    while (right < s.size()) {
        char c1 = s[right];
        if (needs.count(c1)) {
            window[c1]++;
            if (window[c1] == needs[c1]) 
                match++;
        }
        right++;

        while (match == needs.size()) {
            if (right - left < minLen) {
                // Updates the position and length of the smallest string
                start = left;
                minLen = right - left;
            }
            char c2 = s[left];
            if (needs.count(c2)) {
                window[c2]--;
                if (window[c2] < needs[c2])
                    match--;
            }
            left++;
        }
    }
    return minLen == INT_MAX ?
                "" : s.substr(start, minLen);
}

# Find All Anagrams in a String
vector<int> findAnagrams(string s, string t) {
    // Init a collection to save the result
    vector<int> res;
    int left = 0, right = 0;
    // Create a map to save the Characters of the target substring.
    unordered_map<char, int> needs;
    unordered_map<char, int> window;
    for (char c : t) needs[c]++;
    // Maintain a counter to check whether match the target string.
    int match = 0;

    while (right < s.size()) {
        char c1 = s[right];
        if (needs.count(c1)) {
            window[c1]++;
            if (window[c1] == needs[c1])
                match++;
        }
        right++;

        while (match == needs.size()) {
            // Update the result if find a target
            if (right - left == t.size()) {
                res.push_back(left);
            }
            char c2 = s[left];
            if (needs.count(c2)) {
                window[c2]--;
                if (window[c2] < needs[c2])
                    match--;
            }
            left++;
        }
    }
    return res;
}

# Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s) {
    int left = 0, right = 0;
    unordered_map<char, int> window;
    int res = 0; // Record maximum length

    while (right < s.size()) {
        char c1 = s[right];
        window[c1]++;
        right++;
        // If a duplicate character appears in the window
        // Move the left pointer
        while (window[c1] > 1) {
            char c2 = s[left];
            window[c2]--;
            left++;
        }
        res = max(res, right - left);
    }
    return res;
}
'''

## Some Useful Bit Manipulations
'''
# Use OR '|' and space bar coverts English characters to lowercase
('a' | ' ') = 'a'
('A' | ' ') = 'a'

# Use AND '&' and underline coverts English to uppercase
('b' & '_') = 'B'
('B' & '_') = 'B'

# Use XOR '^' and space bar for English characters case exchange
('d' ^ ' ') = 'D'
('D' ^ ' ') = 'd'

# Determine if the sign of two numbers are different
int x = -1, y = 2;
bool f = ((x ^ y) < 0); // true

int x = 3, y = 2;
bool f = ((x ^ y) < 0); // false

# Swap Two Numbers
int a = 1, b = 2;
a ^= b;
b ^= a;
a ^= b;
// a = 2, b = 1

# Plus one
int n = 1;
n = -~n;
// n = 2

# Minus one
int n = 2;
n = ~-n;
// n = 1

# Count Hamming Weight
int hammingWeight(uint32_t n) {
    int res = 0;
    while (n != 0) {
        n = n & (n - 1);
        res++;
    }
    return res;
}

# Determine if a number is an exponent of 2
bool isPowerOfTwo(int n) {
    if (n <= 0) return false;
    return (n & (n - 1)) == 0;
}
'''

## Russian Doll Envelopes Problem
'''
The russian doll envelopes needs to be sorted according to specific rules, and then converted into a Longest Incremental Subsequence Problem.

You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope can fit into another if and only if both the width and height of one envelope is greater than the width and height of the other envelope.
What is the maximum number of envelopes can you Russian doll? (put one inside other)
---
Solution
First sort the width w in ascending order. If you encounter the same situation withw, sort in descending order by height h. Then use all h as an array, and calculate the length of LIS on this array is the answer.
---
// envelopes = [[w, h], [w, h]...]
public int maxEnvelopes(int[][] envelopes) {
    int n = envelopes.length;
    // sort by ascending width, and sort by descending height if the width are the same
    Arrays.sort(envelopes, new Comparator<int[]>() 
    {
        public int compare(int[] a, int[] b) {
            return a[0] == b[0] ? 
                b[1] - a[1] : a[0] - b[0];
        }
    });
    // find LIS on the height array
    int[] height = new int[n];
    for (int i = 0; i < n; i++)
        height[i] = envelopes[i][1];

    return lengthOfLIS(height);
}

/* returns the length of LIS in nums */
public int lengthOfLIS(int[] nums) {
    int piles = 0, n = nums.length;
    int[] top = new int[n];
    for (int i = 0; i < n; i++) {
        // playing card to process
        int poker = nums[i];
        int left = 0, right = piles;
        // position to insert for binary search
        while (left < right) {
            int mid = (left + right) / 2;
            if (top[mid] >= poker)
                right = mid;
            else
                left = mid + 1;
        }
        if (left == piles) piles++;
        // put this playing cart on top of the pile
        top[left] = poker;
    }
    // the number of cards is the LIS length
    return piles;
}
'''
## Recursion 
'''
void sort(Comparable[] a){    
    int N = a.length;
    // So complicated! It shows disrespect for sorting. I refuse to study such code.
    for (int sz = 1; sz < N; sz = sz + sz)
        for (int lo = 0; lo < N - sz; lo += sz + sz)
            merge(a, lo, lo + sz - 1, Math.min(lo + sz + sz - 1, N - 1));
}

/* I prefer recursion, simple and beautiful */
void sort(Comparable[] a, int lo, int hi) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    sort(a, lo, mid); // soft left part
    sort(a, mid + 1, hi); // soft right part
    merge(a, lo, mid, hi); // merge the two sides
}
---
# Divide and conquer algorithm
Merge and sort, typical divide-and-conquer algorithm; divide-and-conquer, typical recursive structure.
The divide-and-conquer algorithm can go in three steps: decomposition-> solve-> merge
Decompose the original problem into sub-problems with the same structure.
After decomposing to an easy-to-solve boundary, perform a recursive solution.
Combine the solutions of the subproblems into the solutions of the original problem.
'''

## Backtracking Solve Subset/Permutation/Combination
'''
# Subset
The first solution is using the idea of mathematical induction: if you have already known the subset of [1,2], can you derive the subset of [1,2,3]? Let's take a look of the subset of [1,2]. 
[ [],[1],[2],[1,2] ]
You will find such a rule:
subset([1,2,3]) - subset([1,2])
= [3],[1,3],[2,3],[1,2,3]
This is a typical recursive structure: The subset of[1,2,3]can be derived by[1,2], and the subset of [1,2] can be derived by [1]. Obviously, the base case is that when the input set is an empty set, the output subset is also an empty set.

vector<vector<int>> subsets(vector<int>& nums) {
    // base case, return an empty set
    if (nums.empty()) return {{}};
    // take the last element
    int n = nums.back();
    nums.pop_back();
    // recursively calculate all subsets of the previous elements
    vector<vector<int>> res = subsets(nums);

    int size = res.size();
    for (int i = 0; i < size; i++) {
        // then append to the previous result
        res.push_back(res[i]);
        res.back().push_back(n);
    }
    return res;
}
---
The second general method is the backtracking algorithm

vector<vector<int>> res;

vector<vector<int>> subsets(vector<int>& nums) {
    // record the path
    vector<int> track;
    backtrack(nums, 0, track);
    return res;
}

void backtrack(vector<int>& nums, int start, vector<int>& track) {
    res.push_back(track);
    for (int i = start; i < nums.size(); i++) {
        // select
        track.push_back(nums[i]);
        // backtrack
        backtrack(nums, i + 1, track);
        // deselect
        track.pop_back();
    }
}
---
# Combination
vector<vector<int>>res;

vector<vector<int>> combine(int n, int k) {
    if (k <= 0 || n <= 0) return res;
    vector<int> track;
    backtrack(n, k, 1, track);
    return res;
}

void backtrack(int n, int k, int start, vector<int>& track) {
    // reach the bottom of tree
    if (k == track.size()) {
        res.push_back(track);
        return;
    }
    // note: i is incremented from start 
    for (int i = start; i <= n; i++) {
        // select
        track.push_back(i);
        backtrack(n, k, i + 1, track);
        // deselect
        track.pop_back();
    }
}
---
# Permutation
List<List<Integer>> res = new LinkedList<>();

/* main function, input a uique set of numbers and return all permutations of them */
List<List<Integer>> permute(int[] nums) {
    // record "path"
    LinkedList<Integer> track = new LinkedList<>();
    backtrack(nums, track);
    return res;
}

void backtrack(int[] nums, LinkedList<Integer> track) {
    // trigger the ending condition
    if (track.size() == nums.length) {
        res.add(new LinkedList(track));
        return;
    }

    for (int i = 0; i < nums.length; i++) {
        // exclud illegal selections
        if (track.contains(nums[i]))
            continue;
        // select
        track.add(nums[i]);
        // go to the next decision tree
        backtrack(nums, track);
        // deselect
        track.removeLast();
    }
}
'''

## Shuffle Algorithm
'''
https://labuladong.gitbook.io/algo-en/iii.-algorithmic-thinking/shuffle_algorithm
'''

## Several counter-intuitive Probability Problems
'''
https://labuladong.gitbook.io/algo-en/iii.-algorithmic-thinking/several_counter_intuitive_probability_problems
'''

### *** High Frequency Interview Problem
## How to Find Prime Number Efficiently
'''
# Inefficient
int countPrimes(int n) {
    int count = 0;
    for (int i = 2; i < n; i++)
        if (isPrim(i)) count++;
    return count;
}

// Determines whether integer n is prime
boolean isPrime(int n) {
    for (int i = 2; i < n; i++)
        if (n % i == 0)
            // There are other divisibility factors
            return false;
    return true;
}

# Efficient implementation countPrimes
int countPrimes(int n) {
    boolean[] isPrim = new boolean[n];
    Arrays.fill(isPrim, true);
    for (int i = 2; i * i < n; i++) 
        if (isPrim[i]) 
            for (int j = i * i; j < n; j += i) 
                isPrim[j] = false;

    int count = 0;
    for (int i = 2; i < n; i++)
        if (isPrim[i]) count++;

    return count;
}
'''

## How to Solve Drop Water Problem
'''
# Brute force
int trap(vector<int>& height) {
    int n = height.size();
    int ans = 0;
    for (int i = 1; i < n - 1; i++) {
        int l_max = 0, r_max = 0;
        // find the highest column on the right
        for (int j = i; j < n; j++)
            r_max = max(r_max, height[j]);
        // find the highest column on the right
        for (int j = i; j >= 0; j--)
            l_max = max(l_max, height[j]);
        // if the position i itself is the highest column
        // l_max == r_max == height[i]
        ans += min(l_max, r_max) - height[i];
    }
    return ans;
}

# Memorized 
int trap(vector<int>& height) {
    if (height.empty()) return 0;
    int n = height.size();
    int ans = 0;
    // arrays act the memo
    vector<int> l_max(n), r_max(n);
    // initialize base case
    l_max[0] = height[0];
    r_max[n - 1] = height[n - 1];
    // calculate l_max from left to right
    for (int i = 1; i < n; i++)
        l_max[i] = max(height[i], l_max[i - 1]);
    // calculate r_max from right to left
    for (int i = n - 2; i >= 0; i--) 
        r_max[i] = max(height[i], r_max[i + 1]);
    // calculate the final result
    for (int i = 1; i < n - 1; i++) 
        ans += min(l_max[i], r_max[i]) - height[i];
    return ans;
}

# Two pointers
int trap(vector<int>& height) {
    if (height.empty()) return 0;
    int n = height.size();
    int left = 0, right = n - 1;
    int ans = 0;

    int l_max = height[0];
    int r_max = height[n - 1];

    while (left <= right) {
        l_max = max(l_max, height[left]);
        r_max = max(r_max, height[right]);

        // ans += min(l_max, r_max) - height[i]
        if (l_max < r_max) {
            ans += l_max - height[left];
            left++; 
        } else {
            ans += r_max - height[right];
            right--;
        }
    }
    return ans;
}
'''

## How to Remove Duplicate From Sorted Sequence
'''
For the array related algorithm problem,there is a general technique: try to avoid deleting the element in the middle, then I want to find a way to swap the element to the last.
---
int removeDuplicates(int[] nums) {
    int n = nums.length;
    if (n == 0) return 0;
    int slow = 0, fast = 1;
    while (fast < n) {
        if (nums[fast] != nums[slow]) {
            slow++;
            // Maintain no repetition of nums[0..slow] 
            nums[slow] = nums[fast];
        }
        fast++;
    }
    //The length is index + 1 
    return slow + 1;
}
---
# Remove Duplicates from Sorted List
ListNode deleteDuplicates(ListNode head) {
    if (head == null) return null;
    ListNode slow = head, fast = head.next;
    while (fast != null) {
        if (fast.val != slow.val) {
            // nums[slow] = nums[fast];
            slow.next = fast;
            // slow++;
            slow = slow.next;
        }
        // fast++
        fast = fast.next;
    }
    // The list disconnects from the following repeating elements
    slow.next = null;
    return head;
}
'''

## How to Find Longest Palindromic Substring
'''
Palindrome string could be in either odd length or even length, a good solution would be double pointers. 
Core idea: start a scanner from the mid point of the string. 
Pseudo code:
for 0 <= i < len(s):
    find a palindrome that set s[i] as its mid point
    find a palindrome that set s[i] and s[i + 1] as its mid point
    update the answer

---
string longestPalindrome(string s) {
    string res;
    for (int i = 0; i < s.size(); i++) {
        // find a palindrome that set s[i] as its mid 
        string s1 = palindrome(s, i, i);
        // find a palindrome that set s[i] and s[i + 1] as its mid  
        string s2 = palindrome(s, i, i + 1);
        // res = longest(res, s1, s2)
        res = res.size() > s1.size() ? res : s1;
        res = res.size() > s2.size() ? res : s2;
    }
    return res;
}
---
Time complexity: O(N^2)
Space complexity: O(1)
By the way, a dynamic programming approach can also work in this problem in a same time complexity. However, we need at least O(N^2) spaces to store DP table. Therefore, in this problem, dp approach is not the best solution.
In addition, Manacher's Algorithm requires only O(N) time complexity. 
'''

## Reverse Linked List in K Group
'''
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
Given this linked list: 1->2->3->4->5 For k = 2, you should return: 2->1->4->3->5 For k = 3, you should return: 3->2->1->4->5
---
Linked list is a kind of data structure with recursion and iteration. On second thought, we can find that this problem can be solved by recursion.

ListNode reverseKGroup(ListNode head, int k) {
    if (head == null) return null;
    // interval [a, b) includes k nodes to be reversed
    ListNode a, b;
    a = b = head;
    for (int i = 0; i < k; i++) {
        // base case
        if (b == null) return head;
        b = b.next;
    }
    // reverse first k nodes
    ListNode newHead = reverse(a, b);
    // merge all reversed internals
    a.next = reverseKGroup(b, k);
    return newHead;
}
'''

## Validation of Parenthesis
'''
Having left parenthesis into stack, as for right parenthesis, find the recent left parenthesis in the stack, and then check if matched.
---
bool isValid(string str) {
    stack<char> left;
    for (char c : str) {
        if (c == '(' || c == '{' || c == '[')
            left.push(c);
        else // character c is right parenthesis
            if (!left.empty() && leftOf(c) == left.top())
                left.pop();
            else
                // not match with recent left parenthesis
                return false;
    }
    // whether all left parenthesis are matched
    return left.empty();
}

char leftOf(char c) {
    if (c == '}') return '{';
    if (c == ')') return '(';
    return '[';
}
'''

## Find Missing Element
'''
This question is not hard. It's easy to think aabout traversing after sorting. Alternatively, using a HashSet to store all the existing elements, and then go through elements in [0, n] and loop up in the HashSet. Both ways can find the correct answer.
However, the time complexity for the sorting solution is O(NlogN). The HashSet solution has O(N) for time complexity, but requires O(N) space complexity to store the data.
---
Bit Manipulation
The XOR operation (^) has a special property: the result of a number XOR itself is 0, and the result of a number with 0 is itself.
How to find out the missing number? Perform XOR operations to all elements and their indices respectively. A pair of an element and its index will become 0. Only the missing element will be left.

int missingNumber(int[] nums) {
    int n = nums.length;
    int res = 0;
    // XOR with the new index first
    res ^= n;
    // XOR with the all elements and the other indices
    for (int i = 0; i < n; i++)
        res ^= i ^ nums[i];
    return res;
}
---
There is actually an even easier solution: Summation of Arithmetic Progression (AP).
int missingNumber(int[] nums) {
    int n = nums.length;
    // Formula: (head + tail) * n / 2
    int expect = (0 + n) * (n + 1) / 2;

    int sum = 0;
    for (int x : nums) 
        sum += x;
    return expect - sum;
---
To avoid overflow, why not perform subtraction while summing up? Similar to our bit operation solution just now, assume nums = [0,3,1,4], add an index such that elements will be paired up with indices respectively.
public int missingNumber(int[] nums) {
    int n = nums.length;
    int res = 0;
    // Added index
    res += n - 0;
    // Summing up the differences between the remaining indices and elements
    for (int i = 0; i < n; i++) 
        res += i - nums[i];
    return res;
}
'''

## Pick Elements From a Arbitrary Sequence
'''
Given a linked which length is unknown, and you need design an algorithm to return one node from the linked list with traversaling the linked list only once.
The simple idea is to firstly traversal the whole linked list and then get the total length n. After that, generate an index from the random number in range [1, n]. Finding the corresponding node of the index means we have found the randomly selected node.
However, the requirement is, traversaling the linked list only once, but such kind of ideas would not fulfill it.
If you want to solve such kind of questions, then you need to learn the Reservoir Sampling algorithm.
---
/* return the value of a random node from the linked list */
int getRandom(ListNode head) {
    Random r = new Random();
    int i = 0, res = 0;
    ListNode p = head;
    // while iterate through the linked list
    while (p != null) {
        // generate an integer in range [0, i) 
        // the possibility of the integer equals to 0 is 1/i
        if (r.nextInt(++i) == 0) {
            res = p.val;
        }
        p = p.next;
    }
    return res;
}
---
/* return the values of k random nodes from the linked list */
int[] getRandom(ListNode head, int k) {
    Random r = new Random();
    int[] res = new int[k];
    ListNode p = head;

    // select first k elements by default
    for (int j = 0; j < k && p != null; j++) {
        res[j] = p.val;
        p = p.next;
    }

    int i = k;
    // while iterate the linked list
    while (p != null) {
        // generate an integer in range [0, i) 
        int j = r.nextInt(++i);
        // the possibility of the integer less than k is k/i
        if (j < k) {
            res[j] = p.val;
        }
        p = p.next;
    }
    return res;
}
---
The time complexity of above sampling algorithm is O(n), but it's not the most optimized method. The better algorithm is based on geometric distribution. The time complexity is O(k + klog(n/k)). 
'''

## Binary Search
'''
KoKo Banana
Koko loves to eat bananas. There are N piles of bananas, the i-th pile has piles[i] bananas. The guards have gone and will come back in H hours.
Koko can decide her bananas-per-hour eating speed of K. Each hour, she chooses some pile of bananas, and eats K bananas from that pile. If the pile has less than K bananas, she eats all of them instead, and won't eat any more bananas during this hour.
Koko likes to eat slowly, but still wants to finish eating all the bananas before the guards come back.
Return the minimum integer K such that she can eat all the bananas within H hours.
int minEatingSpeed(int[] piles, int H) {
    // apply the algorithms framework for searching the left boundary
    int left = 1, right = getMax(piles) + 1;
    while (left < right) {
        // prevent overflow
        int mid = left + (right - left) / 2;
        if (canFinish(piles, mid, H)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

// Time complexity O(N)
boolean canFinish(int[] piles, int speed, int H) {
    int time = 0;
    for (int n : piles) {
        time += timeOf(n, speed);
    }
    return time <= H;
}

int timeOf(int n, int speed) {
    return (n / speed) + ((n % speed > 0) ? 1 : 0);
}

int getMax(int[] piles) {
    int max = 0;
    for (int n : piles)
        max = Math.max(n, max);
    return max;
}
---
Transport problem
The i-th package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.
Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within D days.
// find the left boundary using binary search
int shipWithinDays(int[] weights, int D) {
    // minimum possible load
    int left = getMax(weights);
    // maximum possible load + 1
    int right = getSum(weights) + 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canFinish(weights, D, mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

// If the load is cap, can I ship the goods within D days？
boolean canFinish(int[] w, int D, int cap) {
    int i = 0;
    for (int day = 0; day < D; day++) {
        int maxCap = cap;
        while ((maxCap -= w[i]) >= 0) {
            i++;
            if (i == w.length)
                return true;
        }
    }
    return false;
}
'''
## Scheduling Seats
'''
https://labuladong.gitbook.io/algo-en/iv.-high-frequency-interview-problem/seatscheduling
---
// Map endpoint p to the segment with P as the left endpoint
private Map<Integer, int[]> startMap;
// Map endpoint p to the segment with P as the right endpoint
private Map<Integer, int[]> endMap;
// According to their length, store all line segments from small to large 
private TreeSet<int[]> pq;
private int N;

public ExamRoom(int N) {
    this.N = N;
    startMap = new HashMap<>();
    endMap = new HashMap<>();
    pq = new TreeSet<>((a, b) -> {
        // Calculate the length of two line segments
        int distA = distance(a);
        int distB = distance(b);
        // Longer means it is bigger, and put it back
        return distA - distB;
    });
    // Firstly, put a virtual segment in the ordered set
    addInterval(new int[] {-1, N});
}

/* Remove a line segment */
private void removeInterval(int[] intv) {
    pq.remove(intv);
    startMap.remove(intv[0]);
    endMap.remove(intv[1]);
}

/* Add a line segment */
private void addInterval(int[] intv) {
    pq.add(intv);
    startMap.put(intv[0], intv);
    endMap.put(intv[1], intv);
}

/* Calculate the length of a line segment */
private int distance(int[] intv) {
    return intv[1] - intv[0] - 1;
}

public int seat() {
    // Take the longest line from the ordered set
    int[] longest = pq.last();
    int x = longest[0];
    int y = longest[1];
    int seat;
    if (x == -1) { // case 1
        seat = 0;
    } else if (y == N) { // case 2
        seat = N - 1;
    } else { // case 3
        seat = (y - x) / 2 + x;
    }
    // Divide the longest line segment into two segments
    int[] left = new int[] {x, seat};
    int[] right = new int[] {seat, y};
    removeInterval(longest);
    addInterval(left);
    addInterval(right);
    return seat;
}

public void leave(int p) {
    // Find out the lines around p
    int[] right = startMap.get(p);
    int[] left = endMap.get(p);
    // Merge two segments into one
    int[] merged = new int[] {left[0], right[1]};
    removeInterval(left);
    removeInterval(right);
    addInterval(merged);
}
'''

## Union Find
'''
https://labuladong.gitbook.io/algo-en/iv.-high-frequency-interview-problem/union-find-explanation

The algorithm has three key points:
Use the parent array to record the parent node of each node, which is equivalent to a pointer to the parent node, so theparent array actually stores a forest (several multi-trees).
Use the size array to record the weight of each tree. The purpose is to keep theunion tree still balanced without degrading it into a linked list, which affects the operation efficiency.
Path compression is performed in the find function to ensure that the height of any tree is kept constant, so that the time complexity of theunion and connected API is O (1).
Some readers may ask, Since the path compression, does the weight balance of the size array still need? This problem is very interesting, because path compression guarantees that the tree height is constant (not more than 3), even if the tree is unbalanced, the height is also constant, which basically has little effect.
---
class UF {
    // Number of connected components
    private int count;
    // Store a tree
    private int[] parent;
    // Record the "weight" of the tree
    private int[] size;

    public UF(int n) {
        this.count = n;
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    public void union(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ)
            return;

        // The small tree is more balanced under the big tree
        if (size[rootP] > size[rootQ]) {
            parent[rootQ] = rootP;
            size[rootP] += size[rootQ];
        } else {
            parent[rootP] = rootQ;
            size[rootQ] += size[rootP];
        }
        count--;
    }

    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }

    private int find(int x) {
        while (parent[x] != x) {
            // Path compression
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    public int count() {
        return count;
    }
}
'''

## Union-Find Application
'''
# DFS Alternatives
Many problems solved by the DFS depth-first algorithm can also be solved by the Union-Find algorithm.
For instance, Surrounded Regions of question 130: Given a 2D board containing X and O (the letter O), capture all regions surrounded by X.
void solve(char[][] board) {
    if (board.length == 0) return;

    int m = board.length;
    int n = board[0].length;
    // Leave an extra room for dummy
    UF uf = new UF(m * n + 1);
    int dummy = m * n;
    // Connect the first and last columns of O and dummy
    for (int i = 0; i < m; i++) {
        if (board[i][0] == 'O')
            uf.union(i * n, dummy);
        if (board[i][n - 1] == 'O')
            uf.union(i * n + n - 1, dummy);
    }
    // Connect O and dummy in the first and last rows
    for (int j = 0; j < n; j++) {
        if (board[0][j] == 'O')
            uf.union(j, dummy);
        if (board[m - 1][j] == 'O')
            uf.union(n * (m - 1) + j, dummy);
    }
    // Direction array d is a common method for searching up, down, left and right
    int[][] d = new int[][]{{1,0}, {0,1}, {0,-1}, {-1,0}};
    for (int i = 1; i < m - 1; i++) 
        for (int j = 1; j < n - 1; j++) 
            if (board[i][j] == 'O')
                // Connect this O with up, down, left and right O
                for (int k = 0; k < 4; k++) {
                    int x = i + d[k][0];
                    int y = j + d[k][1];
                    if (board[x][y] == 'O')
                        uf.union(x * n + y, i * n + j);
                }
    // All O not connected to dummy shall be replaced
    for (int i = 1; i < m - 1; i++) 
        for (int j = 1; j < n - 1; j++) 
            if (!uf.connected(dummy, i * n + j))
                board[i][j] = 'X';
}
---
Satisfiability of Equality Equations
Given an array equations of strings that represent relationships between variables, each string equations[i] has length 4 and takes one of two different forms: "a==b" or "a!=b". Here, a and b are lowercase letters (not necessarily different) that represent one-letter variable names.
Return true if and only if it is possible to assign integers to variable names so as to satisfy all the given equations.
The core idea of solving the problem is that divide the expressions in equations into two parts according to == and !=, First process the expressions of ==, so that they are connected. != Expression to check if the inequality relationship breaks the connectivity of the equality relationship.

boolean equationsPossible(String[] equations) {
    // 26 letters
    UF uf = new UF(26);
    // Let equal letters form connected components first
    for (String eq : equations) {
        if (eq.charAt(1) == '=') {
            char x = eq.charAt(0);
            char y = eq.charAt(3);
            uf.union(x - 'a', y - 'a');
        }
    }
    // Check if inequality relationship breaks connectivity of equal relationship
    for (String eq : equations) {
        if (eq.charAt(1) == '!') {
            char x = eq.charAt(0);
            char y = eq.charAt(3);
            // If the equality relationship holds, it is a logical conflict
            if (uf.connected(x - 'a', y - 'a'))
                return false;
        }
    }
    return true;
}
'''
## Find Subsequence With Binary Search
'''
how to determine if a given string s is subsequence of another string t (assume s is much shorter as compared to t)?
int m = s.length(), n = t.length();
ArrayList<Integer>[] index = new ArrayList[256];
// record down the indices of each character in t
for (int i = 0; i < n; i++) {
    char c = t.charAt(i);
    if (index[c] == null) 
        index[c] = new ArrayList<>();
    index[c].add(i);
}
---
boolean isSubsequence(String s, String t) {
    int m = s.length(), n = t.length();
    // pre-process t
    ArrayList<Integer>[] index = new ArrayList[256];
    for (int i = 0; i < n; i++) {
        char c = t.charAt(i);
        if (index[c] == null) 
            index[c] = new ArrayList<>();
        index[c].add(i);
    }

    // the pointer in t
    int j = 0;
    // find s[i] using index
    for (int i = 0; i < m; i++) {
        char c = s.charAt(i);
        // character c does not exist in t
        if (index[c] == null) return false;
        int pos = left_bound(index[c], j);
        // c is not found in the binary search interval
        if (pos == index[c].size()) return false;
        // increment pointer j
        j = index[c].get(pos) + 1;
    }
    return true;
}
'''
## Problems can be solved by one line
'''
# Nim Game
The game rule is that there is a heap of stones on the table for you and friends to remove. Each of you takes turns to remove the stones and can take at least one and at most three each time. The one who takes the last stone will win the game. S

We usually use contrarian thinking to find a solution of this kind of problems：
If I win the game, I need to take the remaining stones (1\~3 stones) at once.
How to make this situation come into being? If there are 4 stones remaining when your opponent takes the chance to pick the stones, no matter how he takes the stones, you can always win the game because there will always be 1~3 stones remaining.
And how to force your opponent to face the situation when there are 4 stones left? If there are 5~7 stones remaining by the time you take your turn, you can let your opponent face 4-stone situation.
Then how to get into a 5~7 stones situation when you are picking? Let your opponent face 8 stones. No matter how he plans to take the stones, we can win the game because of the remaining 5~7 stones.
And so on, we can find out that if n is a multiple of 4, you will fall into the trap and can never win the game.

bool canWinNim(int n) {
    // If n is a multiple of 4, then return false
    // Otherwise, return true
    return n % 4 != 0;
}
---
# Stone Game
The game rule is that you and your friend play a game with piles of stones. The piles of stone are represented by an array, piles. pile[i] refers to the number of stones in the ith pile. Each turn, a player takes the entire pile of stones from either the beginning or the end of the row. And the winner is the one who gets more stones in the end. 

????
---
# Bulb Switcher
there are n bulbs in a room and they are initially turned off. Now we need to do n operations:
- Flip all the lights.
- Flip lights with even numbers.
- Flip the bulb whose number is a multiple of 3 (e.g. 3, 6, 9, ... and 3 is off while 6 is on).
For the i-th round, you toggle every i bulb. For the n-th round, you only toggle the last bulb.
You need to find how many bulbs are on after n rounds.

Suppose we have 16 lights, and we take the square root of 16, which is equal to 4, and that means we're going to end up with 4 lights on. The lights are 1*1=1, 2*2=4, 3*3=9, and 4*4=16.
Some square root of n turns out to be a decimal. However, converting them to integers is the same thing as getting all the integers smaller than a certain integer upper bound, and the square roots of these numbers are the index of the lights on at last. so just turn the square root into an integer, that's the answer to the question.

int bulbSwitch(int n) {
    return (int)Math.sqrt(n);
}
'''

## How to Find Duplicate and Missing Element
'''
The set Soriginally contains numbers from 1 to n. But unfortunately, due to the data error, one of the numbers in the set got duplicate to another number in the set, which results in repetition of one number and loss of another number.
Given an array nums representing the data status of this set after the error. Your task is to firstly find the number occurs twice and then find the number that is missing. Return them in the form of an array.

Firstly, traverse over the whole nums array and use HashMap to store the number of times each element of the array. After this, we can consider every number from 1 to n, and check for its presence in map.
But here's a problem. This solution requires a HashMap that means the space complexity is O(n). We check the condition again. Consider the numbers from 1 to n, which happens to be one duplicate element and one missing element. There must be something strange about things going wrong.
We must traverse over the whole nums array of size n for each of the numbers from 1 to n. That means the time complexity is O(n). So we can think how to save the space used to reduce the space complexity to O(1).

The key point is that elements and indexes appear in pairs for this kind of problems. Common methods include Sorting, XOR, and Map
The idea of Map is the above analysis. Mapping each index and element, and recording whether an element is mapped with a sign.
The Sorting method is also easy to understand. For this problem, we can assume that if all elements are sorted from smallest to largest. If we find that the corresponding elements of the index didn't match, so we find duplicate and missing elements.
XOR operation is also commonly used. The XOR operation (^) has a special property: the result of a number XOR itself is 0, and the result of a number with 0 is itself. For instance: a ^ a = 0, a ^ 0 = a. If we take XOR of the index and element at the same time, the paired index and element can be eliminated, and the remaining are duplicate or missing elements. 

vector<int> findErrorNums(vector<int>& nums) {
    int n = nums.size();
    int dup = -1;
    for (int i = 0; i < n; i++) {
        // Now, elements  start at 1
        int index = abs(nums[i]) - 1;
        // nums[index] < 0  means find the duplicate element
        if (nums[index] < 0)
            dup = abs(nums[i]);
        else
            nums[index] *= -1;
    }

    int missing = -1;
    for (int i = 0; i < n; i++)
        // nums[i] > 0 means find the missing element
        if (nums[i] > 0)
            // Convert index to element
            missing = i + 1;

    return {dup, missing};
}
'''
## How to Check Palindromic LinkedList
'''
The core concept to FIND the palindromic strings is expanding from the middle to the edges.
string palindrome(string& s, int l, int r) {
    // to prevent the indexes from getting out of range
    while (l >= 0 && r < s.size()
            && s[l] == s[r]) {
        // expand to two edges
        l--; r++;
    }
    // return the longest palindromic in which the middle
    // are both s[l] and s[r]
    return s.substr(l + 1, r - l - 1);
}

But to CHECK a palindromic string is much easier. Regardless of its length, we only need to do the double pointers trick, and move from two edges to the middle.
bool isPalindrome(string s) {
    int left = 0, right = s.length - 1;
    while (left < right) {
        if (s[left] != s[right])
            return false;
        left++; right--;
    }
    return true;
}
---
Check A Palindromic Singly Linked List
What is the essence of this way? It is all about pushing the nodes in the linked list into a stack and then popping them out. At this time the elements are in reverse. What we make in use is the queues and stacks in recursion.

// The left pointer
ListNode left;

boolean isPalindrome(ListNode head) {
    left = head;
    return traverse(head);
}

boolean traverse(ListNode right) {
    if (right == null) return true;
    boolean res = traverse(right.next);
    // code to traverse in postorder
    res = res && (right.val == left.val);
    left = left.next;
    return res;
}

---
Optimizing the Space Complexity
Find the node in the middle by the fast and slow pointers:
ListNode slow, fast;
slow = fast = head;
while (fast != null && fast.next != null) {
    slow = slow.next;
    fast = fast.next.next;
}
// the slow pointer now points to the middle point

If the fast pointer doesn't point to null, the length of this linked list is odd, which means the slow pointer needs to forward one more step:
if (fast != null)
    slow = slow.next;

Reverse the right half of the linked list and compare palindromes:
ListNode left = head;
ListNode right = reverse(slow);

while (right != null) {
    if (left.val != right.val)
        return false;
    left = left.next;
    right = right.next;
}
return true;

ListNode reverse(ListNode head) {
    ListNode pre = null, cur = head;
    while (cur != null) {
        ListNode next = cur.next;
        cur.next = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}

p.next = reverse(q);
'''