from collections import Counter, defaultdict, deque


def twoSum(nums, target):
    prevMap = {}

    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i

def maxProfit(prices):
    res = 0

    lowest = prices[0]
    for price in prices:
        if price < lowest:
            lowest = price
        res = max(res, price - lowest)
    return res

def containsDuplicate(nums):
    hashset = set()

    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False

def productExceptSelf(nums):
    res = [1] * (len(nums))

    prefix = 1
    for i in range(len(nums)):
        res[i] = prefix
        prefix *= nums[i]
    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]
    return res

def maxSubArray(nums):
    res = nums[0]

    total = 0
    for n in nums:
        total += n
        res = max(res, total)
        if total < 0:
            total = 0
    return res

def maxProduct(nums):
    res = nums[0]
    curMin, curMax = 1, 1

    for n in nums:
        tmp = curMax * n
        curMax = max(tmp, n * curMin, n)
        curMin = min(tmp, n * curMin, n)
        res = max(res, curMax) 
    return res

def findMin(nums): #AGAIN
    start, end = 0, len(nums) - 1
    curr_min = float('inf')

    while start < end:
        mid = (start + end) // 2
        curr_min = min(curr_min, nums[mid])

        if nums[mid] > nums[end]:
            start = mid + 1
        else:
            end = mid - 1
    return min(curr_min, nums[start])

def searchRotated(nums, target):
    l, r = 0, len(nums) - 1

    while l <= r:
        mid = (l + r) // 2
        if target == nums[mid]:
            return mid
        if nums[l] <= nums[mid]:
            if target > nums[mid] or target < nums[l]:
                l = mid + 1
            else:
                r = mid - 1
        else:
            if target < nums[mid] or target > nums[r]:
                r = mid - 1
            else:
                l = mid + 1

    return -1

def threeSum(nums):
    res = []
    nums.sort()

    for i, a in enumerate(nums):
        if a > 0:
            continue
        if i > 0 and a == nums[i - 1]:
            continue
        
        l, r = i + 1, len(nums) - 1
        while l < r:
            threeSum = a + nums[l] + nums[r]
            if threeSum > 0:
                r -= 1
            elif threeSum < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                l += 1
                r -= 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1

def maxContainer(heights):
    l, r = 0, len(heights) - 1
    res = 0

    while l < r:
        res = max(res, min(heights[l], heights[r]) * (r - l))
        if heights[l] < heights[r]:
            l += 1
        elif heights[r] <= heights[l]:
            r -= 1
    return res

def hammingWeight(n):
    res = 0
    while n:
        n &= n - 1
        res += 1
    return res

def reverseBits(n):
    res = 0
    for i in range(32):
        bit = (n >> i) & 1
        res += (bit << (31 - i))
    return res

def climbStairs(n):
    if n <= 3:
        return n
    n1, n2 = 2, 3

    for i in range(4, n + 1):
        tmp = n1 + n2
        n1 = n2
        n2 = tmp
    return n2

def coinChange(coins, amount):
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount + 1):
        for c in coins:
            if a - c >= 0:
                dp[a] = min(dp[a], 1 + dp[a - c])
    return dp[amount] if dp[amount] != amount + 1 else -1

def lengthOfLIS(nums):
    LIS = [1] * len(nums)

    for i in range(len(nums) - 1, -1, -1):
        for j in range(i + 1, len(nums)):
            LIS[i] = max(LIS[i], 1 + LIS[i])
    return max(LIS)

def longestCommonSubsequence(text1, text2):
    dp  = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]

    for i in range(len(text1) - 1, -1, -1):
        for j in range(len(text2) - 1, -1, -1):
            if text1[i] == text2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[0][0]

def wordBreak(s, wordDict):
    dp = [False] * (len(s) + 1)
    dp[len(s)] = True

    for i in range(len(s) - 1, -1, -1):
        for w in wordDict:
            if (i + len(w)) <= len(s) and s[i:i + len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    return dp[0]

def combinationSum(candidates, target):
    res = []

    def dfs(i, curr, total):
        if total == target:
            res.append(curr.copy())
            return
        if i >= len(candidates) or total > target:
            return
        
        curr.append(candidates[i])
        dfs(i, curr, total + candidates[i])
        curr.pop()
        dfs(i + 1, curr, total)
    dfs(0, [], 0)
    return res

def rob(nums):
    rob1, rob2 = 0, 0

    for n in nums:
        temp = max(n + rob1, rob2)
        rob1 = rob2
        rob2 = temp
    return rob2

def rob2(nums):
    return max(nums[0], helperRob2(nums[1:]), helperRob2[:-1])

def helperRob2(nums):
    rob1, rob2 = 0, 0

    for n in range(nums):
        tmp = n + rob1
        rob1 = rob2
        rob2 = tmp
    return rob2

def uniquePaths(m, n):
    row = [1] * n

    for i in range(m - 1):
        newRow = [1] * n
        for j in range(n - 2, -1, -1):
            newRow[j] = newRow[j + 1] + row[j]
        row = newRow
    return row[0]

def canJump(nums):
    goal = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= goal:
            goal = i
    return goal == 0

def cloneGraph(node):
    oldToNew = {}

    def dfs(node):
        if node in oldToNew:
            return oldToNew[node]
        
        copy = Node(node.val)
        oldToNew[node] = copy
        for nei in node.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node) if node else None

def pacificAtlantic(heights):
    ROWS, COLS = len(heights), len(heights[0])
    pac, atl = set(), set()

    def dfs(r, c, visit, prevHeight):
        if(
            (r, c) in visit
            or r < 0
            or c < 0
            or r == ROWS
            or c == COLS
            or heights[r][c] < prevHeight
        ):
            return
        visit.add((r, c))
        dfs(r + 1, c, visit, heights[r][c])
        dfs(r - 1, c, visit, heights[r][c])
        dfs(r, c + 1, visit, heights[r][c])
        dfs(r, c - 1, visit, heights[r][c])

    for c in range(COLS):
        dfs(0, c, pac, heights[0][c])
        dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])

    for r in range(ROWS):
        dfs(r, 0, pac, heights[r][0])
        dfs(r, COLS - 1, atl, heights[r][COLS -1])

    res = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in pac and (r, c) in atl:
                res.append([r, c])
    return res

def numIslands(grid):
    if not grid or not grid[0]:
        return 0
    
    islands = 0
    visit = set()
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if (
            r not in range(rows)
            or c not in range(cols)
            or grid[r][c] == "0"
            or (r, c) in visit
        ):
            return
        
        visit.add((r, c))
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for dr, dc in directions:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visit:
                islands += 1
                dfs(r, c)
    
    return islands

def islandsBFS(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    islands = 0

    def bfs(r, c):
        q = deque()
        visited.add((r, c))
        q.append((r, c))

        while q:
            row, col = q.popleft()
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

            for dr, dc in directions:
                r, c = row + dr, col + dc
                if r in range(rows) and c in range(cols) and grid[r][c] == '1' and (r, c) not in visited:
                    q.append((r, c))
                    visited.add((r, c))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                bfs(r, c)
                islands += 1

    return islands

def longestConsecutive(nums):
    numSet = set(nums)
    longest = 0

    for n in nums:
        if (n - 1) not in numSet:
            length = 1
            while (n + length) in numSet:
                length += 1
            longest = max(length, longest)
    return longest
            
class Solution:
    def alienOrder(self, words):
        adj = {char: set() for word in words for char in word} 

        for i in range(len(words) -1):
            w1, w2 = words[i], words[i + 1]
            minLen = min(len(w1), len(w2))           
            if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
                return ""
            for j in range(minLen):
                if w1[j] != w2[j]:
                    print(w1[j], w2[j])
                    adj[w1[j].add(w2[j])]
                    break
        
        visited = {}
        res = []

        def dfs(char):
            if char in visited:
                return visited[char]
            
            visited[char] = True

            if neighChar in adj[char]:
                if dfs(neighChar):
                    return True
            
            visited[char] = False
            res.append(char)

        for char in adj:
            if dfs(char):
                return ""
        
        res.reverse()

        return "".join(res)
    

def validTree(n, edges):
    if not n:
        return True
    adj = {i: [] for i in range(n)}
    for n1, n2 in edges:
        adj[n1].append(n2)
        adj[n2].append(n2)

    visit = set()

    def dfs(i, prev):
        if i in visit:
            return False
        visit.add(i)
        for j in adj[i]:
            if j == prev:
                continue
            if not dfs(j, i):
                return False
        return True
    return dfs(0, -1) and n == len(visit)


class UnionFind:
    def __init__(self):
        self.f = {}

    def findParent(self, x):
        y = self.f.get(x, x)
        if x != y:
            y = self.f[x] = self.findParent(y)
        return y
    
    def union(self, x, y):
        self.f[self.findParent(x)] = self.findParent(y)

def countComponents(n, edges):
    dsu = UnionFind()
    for a, b in edges:
        dsu.union(a, b)
    return len(set(dsu.findParent(x) for x in range(n)))


def insert(intervals, newInterval):
    res = []

    for i in range(len(intervals)):
        if newInterval[1] < intervals[i][0]:
            res.append(newInterval)
            return res + intervals[i:]
        elif newInterval[0] > intervals[i][1]:
            res.append(intervals[i])
        else:
            newInterval = [
                min(newInterval[0], intervals[i][0]),
                max(newInterval[1], intervals[i][1])
            ]
    res.append(newInterval)
    return res

def merge(intervals):
    intervals.sort(key = lambda pair: pair[0])
    output = [intervals[0]]

    for start, end in intervals:
        lastEnd = output[-1][1]
        if start <= lastEnd:
            output[-1][1] = max(lastEnd, end)
        else:
            output.append([start, end])
    return output

def eraseOverlapIntervals(intervals):
    intervals.sort()
    res = 0
    prevEnd = intervals[0][1]
    for start, end in intervals[1:]:
        if start >= prevEnd:
            prevEnd = end
        else:
            res += 1
            prevEnd = min(end, prevEnd)
    return res

def canAttendMeetings(intervals):
    intervals.sort(key = lambda i: i[0])

    for i in range(1, len(intervals)):
        i1 = intervals[i - 1]
        i2 = intervals[i]

        if i1[1] > i2[0]:
            return False
        return True
    
def minMeetingRooms(intervals):
    time = []
    for start, end in intervals:
        time.append((start, 1))
        time.append((end, -1))

        time.sort(key = lambda x: (x[0], x[1]))

    count = 0
    max_count = 0
    for t in time:
        count += t[1]
        max_count = max(max_count, count)
    return max_count

class ListNode:
    def __init__(self, x=None, y=None):
        self.val = x
        self.next = y

def reverseList(head):
    prev, curr = None, head

    while curr:
        tmp = curr.next
        curr.next = prev
        prev = curr
        curr = tmp
    return prev

def hasCycle(head):
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    return False

def mergeTwoLists(list1, list2):
    dummy = ListNode()
    tail = dummy

    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next

    if list1:
        tail.next = list1
    elif list2:
        tail.next = list2
    
    return dummy.next

class SolutionMergeK:
    def mergeKLists(self, lists):
        if not lists or len(lists) == 0:
            return None
        
        while len(lists) > 1:
            mergedList = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                mergedList.append(self.mergeList(l1, l2))
            list = mergedList
        return lists[0]

    def mergeList(self, l1, l2):
        dummy = ListNode()
        tail = dummy

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next

        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        
        return dummy.next
    
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    left = dummy
    right = head

    while n > 0:
        right = right.next
        n -= 1

    while right:
        left = left.next
        right = right.next

    left.next = left.next.next
    return dummy.next

def reorderList(head):
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    second = slow.next
    prev = slow.next = None

    while second:
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp

    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2

def setZeroes(matrix):
    ROWS, COLS = len(matrix), len(matrix[0])
    rowZero = False

    for r in range(ROWS):
        for c in range(COLS):
            if matrix[r][c] == 0:
                matrix[0][c] = 0
                if r > 0:
                    matrix[r][0] = 0
                else:
                    rowZero = True
    
    for r in range(1, ROWS):
        for c in range(1, COLS):
            if matrix[0][c] == 0 or matrix[r][0] == 0:
                matrix[r][c] = 0

    if matrix[0][0] == 0:
        for r in range(ROWS):
            matrix[r][0] = 0

    if rowZero:
        for c in range(COLS):
            matrix[0][c] = 0


def spiralOrder(matrix):
    res = []
    left, right = 0, len(matrix[0])
    top, bottom = 0, len(matrix)

    while left < right and top < bottom:
        for i in range(left, right):
            res.append(matrix[top][i])
        top += 1

        for i in range(top, bottom):
            res.append(matrix[i][right - 1])
        right -= 1

        if not (left < right and top < bottom):
            break
        
        for i in range(right - 1, left - 1, -1):
            res.append(matrix[bottom - 1][i])
        bottom -= 1

        for i in range(bottom - 1,  top - 1, -1):
            res.append(matrix[i][left])

    return res

def rotate(matrix):
    l, r = 0, len(matrix) - 1

    while l < r:
        for i in range(r - l):
            top, bottom = l, r

            topLeft = matrix[top][l + i]
            matrix[top][l + i] = matrix[bottom -1][l]
            matrix[bottom - 1][l] = matrix[bottom][r - i]
            matrix[bottom][r - i] = matrix[top + i][r]
            matrix[top + 1][r] = topLeft
        r -= 1
        l += 1

class SolutionWord:
    def exist(self, board, word):
        ROWS, COLS = len(board), len(board[0])
        path = set()

        def dfs(r, c, i):
            if i == len(word):
                return True
            
            if (
                min(r, c) < 0
                or r >= ROWS
                or c >= COLS
                or word[i] != board[r][c]
                or (r, c) in path
            ):
                return False
            
            path.add((r, c))
            res = (
                dfs(r + 1, c, i + 1)
                or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1)
                or dfs(r, c - 1, i + 1)
            )
            path.remove((r, c))

        count = defaultdict(int, sum(map(Counter, board), Counter()))

        if count[word[0]] > count[word[-1]]:
            word = word[::-1]

        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, 0):
                    return True
        return False
    
def lengthOfLongestSubstring(s):
    charSet = set()
    l = 0
    res = 0

    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
        charSet.ad(s[r])
        res = max(res, r - l + 1)
    return res

def characterReplacement(s, k):
    count = {}
    res = 0

    l = 0
    maxf = 0
    for r in range(len(s)):
        count[s[r]] = 1 + count.get(s[r], 0)
        maxf = max(maxf, count(s[r]))

        if (r - l + 1) - maxf > k:
            count[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1)

    return res

def minWindow(s, t):
    if t == "":
        return ""
    
    countT, window = {}, {}
    for c in t:
        countT[c] = 1 + countT.get(c, 0)

    have, need = 0, len(countT)
    res, resLen = [-1, -1], float("infinity")
    l = 0
    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c, 0)

        if c in countT and window[c] == countT[c]:
            have += 1

        while have == need:
            if (r - l + 1) < resLen:
                res = [l, r]
                resLen = r - l + 1
            window[s[l]] -= 1
            if s[l] in countT and window[s[l]] < countT[s[l]]:
                have -= 1
            l += 1
    l, r = res
    return s[l: r + 1] if resLen != float("infinity") else ""

def isAnagram(s, t):
    if len(s) != len(t):
        return False
    
    countS, countT = {}, {}

    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[s[i]] = 1 + countT.get(s[i], 0)
    return countS == countT

def groupAnagram(strs):
    ans = defaultdict(list)

    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        ans[tuple(count)].append(s)
    return ans.values()

def validParen(s):
    Map = {")": "(", "]": "[", "}": "{"}
    stack = []

    for c in s:
        if c not in Map:
            stack.append(c)
            continue
        if not stack or stack[-1] != Map[c]:
            return False
        stack.pop()
    return not stack

class SolutionPalidrome:
    def isPalindrome(self, s):
        l, r = 0, len(s) - 1
        
        while l < r:
            while l < r and not self.alphanum(s[l]):
                l += 1
            while l < r and not self.alphanum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

    def alphanum(self, c):
        return (
            ord("A") <= ord(c) <= ord("Z")
            or ord("a") <= ord(c) <= ord("z")
            or ord("0") <= ord(c) <= ord("9")
        )

def longestPalindrome(s):
    res = ""
    resLen = 0
    
    for i in range(len(s)):
        l, r = i, i
        while l >= 0 and r <= len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l: r + 1]
                resLend = r - l + 1
            l -= 1
            r += 1

        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l]== s[r]:
            if (r - l + 1) > resLen:
                res = s[l : r + 1]
                resLen = r - l + 1
            l -= 1
            r += 1
    return res

class paliSubstrings:
    def countSubstrings(self, s):
        res = 0

        for i in range(len(s)):
            res += self.countPali(s, i, i)
            res += self.countPali(s, i, i + 1)
        return res
    
    def countPali(self, s, l, r):
        res = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            res += 1
            l -= 1
            r += 1
        return res

def encode(strs):
    res = ""
    for s in strs:
        res += str(len(s)) + "#" + s
    return res

def decode(s):
    res, i = [], 0

    while i < len(s):
        j = i
        while s[j] != "#":
            j += 1
        length = int(s[i:j])
        res.append(s[j + 1 : j + 1 + length])
        i = j + 1 + length
    return res

def maxDepthDFS(root):
    if not root:
        return 0
    return 1 + max(maxDepthDFS(root.left), maxDepthDFS(root.right))

def itermaxDepthDFS(root):
    stack = [[root, 1]]
    res = 0

    while stack:
        node, depth = stack.pop()

        if node:
            res = max(res, depth)
            stack.append([node.left, depth + 1])
            stack.append([node.right, depth + 1])
    return res

def maxDepthBFS(root):
    q = deque()

    if root:
        q.append(root)

    level = 0
    while q:
        for i in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level += 1
    return level

def isSameTree(p, q):
    if not p and not q:
        return True
    if p and q and p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    else:
        return False
    
def invertTree(root):
    if not root:
        return None
    
    tmp = root.left
    root.left = root.right
    root.right = tmp

    invertTree(root.left)
    invertTree(root.right)
    
    return root

def maxPathSum(root):
    res = [root.val]

    def dfs(root):
        if not root:
            return 0
        
        leftMax = dfs(root.left)
        rightMax = dfs(root.right)
        leftMax = max(leftMax, 0)
        rightMax = max(rightMax, 0)

        res[0] = max(res[0], root.val + leftMax + rightMax)
        return root.val + max(leftMax, rightMax)
    dfs(root)
    return res[0]

def levelOrder(root):
    res = []
    q = deque()
    if root:
        q.append(root)

    while q:
        val = []
        for i in range(len(q)):
            node = q.popleft()
            val.append(node)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(val)
    return res

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def serialize(root):
    res = []

    def dfs(node):
        if not node:
            res.append("N")
            return 
        res.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(res)

def deserialize(data):
    vals = data.split(",")
    i = 0

    def dfs():
        if vals[i] == "N":
            i += 1
            return None
        node = TreeNode(int(vals[i]))
        i += 1
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()

def isSubTree(s, t):
    if not t:
        return True
    if not s:
        return False
    if isSameTree(s, t):
        return True
    
    return isSameTree(s.left, t) or isSameTree(s.right, t)


def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = buildTree(preorder[1: mid + 1], inorder[:mid])
    root.right = buildTree(preorder[mid + 1:], inorder[mid + 1:])
    return root

def isValidBST(root):
    def valid(node, left, right):
        if not node:
            return True
        if not (node.val < right and node.val > left):
            return False
        return valid(node.left, left, node.val) and valid(node.right, node.val, right)
    return valid(root, float("-inf"), float("inf"))


def kthSmallest(root, k):
    stack = []
    curr = root

    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop
        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right

def lowestCommonAncestor(root, p, q):
    curr = root
    while curr:
        if p.val > curr.val and q.val > curr.val:
            curr = curr.right
        elif p.val < curr.val and q.val < curr.val:
            curr = curr.left
        else:
            return curr
        
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        curr = self.root
        for c in word:
            i = ord(c) - ord("a")
            if curr.children[i] == None:
                curr.children[i] = TrieNode()
            curr = curr.children[i]    
        curr.end = True

    def search(self, word):
        curr = self.root
        
        for c in word:
            i = ord(c) - ord("a")
            if curr.children[i] == None:
                return False
            curr = curr.children[i]
        return curr.end
    
    def startsWith(self, prefix):
        curr = self.root
        for c in prefix:
            i = ord(c) - ord("a")
            if curr.children[i] == None:
                return False
            curr = curr.children[i]
        return True
    
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[i]
        curr.end = True

    def search(self, word):
        def dfs(j, root):
            curr = root

            for i in range(j, len(word)):
                c = word[i]
                if c == ".":
                    for child in curr.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False
                else:
                    if c not in curr.children:
                        return False
                    curr = curr.children[c]
            return curr.end
        return dfs(0, self.root)

def topKFrequent(nums, k):
    count = {}
    freq = [[] for i in range(len(nums) + 1)]

    for n in nums:
        count[n] = 1 + count.get(n, 0)
    for n, c in count.items():
        freq[c].append(n)
    res = []
    for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res
  

### *** Bit Manipulation
def countDigit(n):
    count = 0
    while n > 0:
        count += 1
        n = n // 10
    return count

def decimalToBinary(n):
    count = 0
    while n > 0:
        count += 1
        n >>= 1
    return count

def decimalToBinary2(n):
    bit_counter = 0
    while  (1 << bit_counter) <= n:
        bit_counter += 1
    return bit_counter

def helperSetBit(n):
    count = 0
    while n > 0:
        count += (n & 1)
        n >>= 1
    return count

def countSetBits(n):
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

def countBitsII(n):
    ans = [0] * (n + 1)
    for i in range(n + 1):
        ans[i] = countSetBits(i)
    return ans

def isEven(array):
    result = []

    def helper(array):
        k = 0
        for n in array:
            result.append("Odd" if (n & 1) == 1 else "Even")
            k += 1
        return result
    return helper(array)

def isPowerOf2(n):
    if n == 0:
        return False
    return (n & (n - 1)) == 0

def minFlip(a, b, c):
    ans = 0
    for i in range(0, 32):
        bitC = ((c >> i) & 1)
        bitA = ((a >> i) & 1)
        bitB = ((a >> i) & 1)

        if ((bitA | bitB) != bitC):
            if bitC == 0:
                if (bitA == 1 and bitB == 1):
                    ans += 2
                else:
                    ans += 1
            else:
                ans += 1
    return ans

def switchSign(number):
    return ~number + 1

def swapNums(a, b):
    a = a ^ b
    b = b ^ a
    a = a ^ b

def OddOccurence(array):
    res = 0
    for value in array:
        res = res ^ value
    return res

def oppositeSign(x, y):
    return (x ^ y) < 0

def hammingDistance(a, b):
    xor = a ^ b
    distance = 0

    while xor != 0:
        distance += 1
        xor &= (xor - 1)
    return distance

def singleNumber(nums):
    xor = 0
    for num in nums:
        xor ^= num
    return xor

def missingNumberHash(array):
    num_set = set()

    for num in array:
        num+set.add(num)
    
    n = len(array) + 1

    for i in range(n):
        if i not in num_set:
            return i
    return -1

def missingNumber(nums):
    missing = len(nums)
    for i in range(0, len(nums)):
        missing ^= i ^ nums[i]
    return missing

def bitsLength(n):
    bitCounter = 0

    while (1 << bitCounter) <= n:
        bitCounter += 1
    return bitCounter

def checkKthBitSet(n, k):
    return (n & (1 << (k - 1))) != 0

def checkKthBitSetRight(n, k):
    return ((n >> (k - 1)) & 1) == 1

def subsets(nums):
    result = []

    n = len(nums)
    pow_size = 2 ** n

    for i in range(pow_size):
        val = []
        for j in range(n):
            if (i & (1 << j)) != 0:
                val.append(nums[j])
        result.append(val)
    return result

def firstSetBit(n):
    if (n == 0):
        return 0
    
    k = 1
    while True:
        if ((n & (1 << (k - 1))) == 0):
            k += 1
        else:
            return k
        
## DP
def find_knapsack(capacity, weights, values, n):
    if n == 0 or capacity == 0:
        return 0
    
    if (weights[n - 1] <= capacity):
        return max(
            values[n - 1] + find_knapsack(capacity - weights[n - 1], weights, values, n - 1),
            find_knapsack(capacity, weights, values, n - 1)
        )
    else:
        return find_knapsack(capacity, weights, values, n - 1)
    
def find_knapsack_value(capacity, weights, values, n, dp):
    if n == 0 or capacity == 0:
        return 0
    
    if dp[n][capacity] != -1:
        return dp[n][capacity]
    
    if weights[n - 1] <= capacity:
        dp[n][capacity] = max(
            values[n - 1] + find_knapsack_value(capacity - weights[n - 1], weights, values, n - 1, dp),
            find_knapsack_value(capacity, weights, values, n - 1,  dp)
        )
        return dp[n][capacity]

def find_knapsack(capacity, weights, values, n):
    dp = [[-1 for i in range(capacity + 1)] for j in range(n + 1)]
    return find_knapsack_value(capacity, weights, values, n, dp)
        
def find_knapsackBU(capacity, weights, values, n):
    dp = [[0 for i in range(capacity + 1)] for j in range(n + 1)]

    for i in range(len(dp)):
        if i == 0 or j == 0:
            dp[i][j] = 0
        elif weights[i - 1] <= j:
            dp[i][j] = max(
                values[i - 1] + dp[i - 1][j - weights[i - 1]],
                dp[i - i][j]
            )
        else:
            dp[i][j] = dp[i - 1][j]
    return dp[-1][-1]

import math
def min_refuel_stops_helper(index, used, cur_fuel, stations):
    if used == 0:
        return cur_fuel
    if used > index:
        return -math.inf
    
    result1 = min_refuel_stops_helper(index - 1, used, cur_fuel, stations)
    result2 = min_refuel_stops_helper(index - 1, used - 1, cur_fuel, stations)

    result = max(result1, -math.inf if result2 < stations[index - 1][0] else result2 + stations[index - 1][1])
    return result

def min_refuel_stops(target, start_fuel, stations):
    n = len(stations)
    i = 0
    max_d = [-1 for i in range(n + 1)]

    while i <= n:
        max_d[i] = min_refuel_stops_helper(n, i, start_fuel, stations)
        i += 1
    result = -1
    
    i = 0
    while i <= n:
        if max_d[i] >= target:
            result = i
            break
        i += 1
    return result

