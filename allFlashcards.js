const arraysHashingFlashcards = [
{
id: 1,
title: "Contains Duplicate",
question: "Check if an array contains any duplicates.",
hint: "Think about using a data structure that allows for quick lookup.",
oneLiner: "Use a set to check if any element repeats.",
simpleExplanation: "We go through each number.\nWe keep a list of ones we've already seen.\nIf a number shows up again, we know it's a duplicate.",
mnemonics: [
"\"Seen before?\" → if num in seen: return True",
"\"Add new\" → seen.add(num)",
"\"End clean\" → return False"
],
code: `def hasDuplicate(nums):
seen = set()
for num in nums:
if num in seen:
return True
seen.add(num)
return False`,
timeComplexity: "O(n) - one pass through the array",
spaceComplexity: "O(n) - in worst case, all elements in set"
},
{
id: 2,
title: "Valid Anagram",
question: "Determine if two strings are anagrams (contain the same letters).",
hint: "Count the frequency of characters in both strings and compare.",
oneLiner: "Use counters (hashmaps) to compare character frequencies.",
simpleExplanation: "We count the letters in both words.\nIf every letter shows up the same number of times, it's an anagram.\nOtherwise, it's not.",
mnemonics: [
"\"Count both\" → Counter(s) == Counter(t)",
"\"Compare maps\" → for key in counts: compare values",
"\"Return result\" → return True/False"
],
code: `def isAnagram(s, t):
if len(s) != len(t):
return False
count = [0] * 26
for i in range(len(s)):
count[ord(s[i]) - ord('a')] += 1
count[ord(t[i]) - ord('a')] -= 1
for val in count:
if val != 0:
return False
return True`,
timeComplexity: "O(n) - where n is the length of the strings",
spaceComplexity: "O(1) - fixed array size for character counts"
},
{
id: 3,
title: "Two Sum",
question: "Find two numbers in an array that add up to a target.",
hint: "Use a hash map to store values you've seen and their indices.",
oneLiner: "Store each number's complement in a hashmap as we loop.",
simpleExplanation: "We look for two numbers that add to a target.\nWe remember each number we see and what we need to reach the target.\nIf we find the right pair, we return their positions.",
mnemonics: [
"\"Check map\" → if target - num in seen: return [i, seen[target - num]]",
"\"Store index\" → seen[num] = i",
"\"Loop through\" → for i, num in enumerate(nums):"
],
code: `def twoSum(nums, target):
indices = {} # val -> index
for i, n in enumerate(nums):
diff = target - n
if diff in indices:
return [indices[diff], i]
indices[n] = i
return []`,
timeComplexity: "O(n) - one pass through the array",
spaceComplexity: "O(n) - for the hash map"
},
{
id: 4,
title: "Group Anagrams",
question: "Group a list of strings such that all anagrams are together.",
hint: "Use a common representation (like sorted string) as a key to group anagrams.",
oneLiner: "Group words using sorted letters as a key in a hashmap.",
simpleExplanation: "We sort the letters in each word.\nWords with the same letters go into the same group.\nWe collect all the groups.",
mnemonics: [
"\"Sort word\" → key = ''.join(sorted(word))",
"\"Group by key\" → anagram_map[key].append(word)",
"\"Return values\" → return anagram_map.values()"
],
code: `def groupAnagrams(strs):
res = defaultdict(list)
for s in strs:
count = [0] * 26
for c in s:
count[ord(c) - ord('a')] += 1
res[tuple(count)].append(s)
return list(res.values())`,
timeComplexity: "O(n*k) - where n is the number of strings and k is the max string length",
spaceComplexity: "O(n) - for storing all strings"
},
{
id: 5,
title: "Top K Frequent Elements",
question: "Find the k most frequent elements in an array.",
hint: "Use a counter to track frequencies and a heap to find the top k elements.",
oneLiner: "Use a counter and a heap to get the top k items.",
simpleExplanation: "We count how many times each number shows up.\nWe keep the k most common ones.\nThen we return them.",
mnemonics: [
"\"Count frequency\" → count = Counter(nums)",
"\"Heap select top\" → heapq.nlargest(k, count.keys(), key=count.get)",
"\"Return result\" → return result"
],
code: `def topKFrequent(nums, k):
count = {}
freq = [[] for i in range(len(nums) + 1)]
for num in nums:
count[num] = 1 + count.get(num, 0)
for num, cnt in count.items():
freq[cnt].append(num)
res = []
for i in range(len(freq) - 1, 0, -1):
for num in freq[i]:
res.append(num)
if len(res) == k:
return res`,
timeComplexity: "O(n) - bucket sort approach",
spaceComplexity: "O(n) - for the frequency counters"
},
{
id: 6,
title: "Encode and Decode Strings",
question: "Design an algorithm to encode and decode a list of strings into a single string.",
hint: "Use a delimiter with length information to separate strings unambiguously.",
oneLiner: "Use a length prefix or delimiter to encode, then decode safely.",
simpleExplanation: "We turn each word into a number + word combo.\nThat way we can separate them later.\nWe read the length to split them back correctly.",
mnemonics: [
"\"Encode with length\" → s = str(len(word)) + \"#\" + word",
"\"Split by count\" → length = int(s[:i]), then s[i+1:i+1+length]",
"\"Loop until done\" → while i < len(s): decode"
],
code: `class Solution:
def encode(self, strs):
res = ""
for s in strs:
res += str(len(s)) + "#" + s
return res
def decode(self, s):
res = []
i = 0
while i < len(s):
j = i
while s[j] != '#':
j += 1
length = int(s[i:j])
i = j + 1
j = i + length
res.append(s[i:j])
i = j
return res`,
timeComplexity: "O(n) - linear in total string length",
spaceComplexity: "O(n) - for the encoded/decoded strings"
},
{
id: 7,
title: "Product of Array Except Self",
question: "Compute the product of all elements except self without division.",
hint: "Use prefix and suffix products to compute the result in two passes.",
oneLiner: "Use prefix and suffix multiplications without division.",
simpleExplanation: "We find how much every number would be if we multiply all other numbers.\nFirst, we go left to right.\nThen, we go right to left and finish the math.",
mnemonics: [
"\"Left pass\" → for i in range(n): res[i] *= prefix",
"\"Right pass\" → for i in reversed(range(n)): res[i] *= suffix",
"\"Update prefixes\" → prefix *= nums[i], suffix *= nums[i]"
],
code: `def productExceptSelf(nums):
res = [1] * (len(nums))
prefix = 1
for i in range(len(nums)):
res[i] = prefix
prefix *= nums[i]
postfix = 1
for i in range(len(nums) - 1, -1, -1):
res[i] *= postfix
postfix *= nums[i]
return res`,
timeComplexity: "O(n) - two passes through the array",
spaceComplexity: "O(1) - excluding output array"
},
{
id: 8,
title: "Valid Sudoku",
question: "Determine if a 9x9 Sudoku board is valid.",
hint: "Check each row, column, and 3x3 sub-box for duplicates.",
oneLiner: "Use sets to check each row, column, and 3x3 box.",
simpleExplanation: "Each row, column, and box must have unique numbers.\nWe go through the board and record what we see.\nIf anything repeats in the same row, column, or box, it's not valid.",
mnemonics: [
"\"Check each cell\" → for i in range(9): for j in range(9):",
"\"Use 3 keys\" → row[i], col[j], box[i//3, j//3]",
"\"Add or return False\" → if val in set: return False else: add"
],
code: `def isValidSudoku(board):
for row in range(9):
seen = set()
for i in range(9):
if board[row][i] == ".":
continue
if board[row][i] in seen:
return False
seen.add(board[row][i])
for col in range(9):
seen = set()
for i in range(9):
if board[i][col] == ".":
continue
if board[i][col] in seen:
return False
seen.add(board[i][col])
for square in range(9):
seen = set()
for i in range(3):
for j in range(3):
row = (square//3) * 3 + i
col = (square % 3) * 3 + j
if board[row][col] == ".":
continue
if board[row][col] in seen:
return False
seen.add(board[row][col])
return True`,
timeComplexity: "O(1) - fixed size (9x9) board",
spaceComplexity: "O(1) - fixed size sets"
},
{
id: 9,
title: "Longest Consecutive Sequence",
question: "Find the length of the longest consecutive sequence in an unsorted array.",
hint: "Use a set for O(1) lookups and check for sequence starts.",
oneLiner: "Use a set and expand sequences from the smallest number.",
simpleExplanation: "We put all the numbers in a quick lookup set.\nThen we start from numbers that are the beginning of a sequence.\nWe count how long each chain goes.",
mnemonics: [
"\"Add to set\" → num_set = set(nums)",
"\"Start from beginning\" → if num - 1 not in num_set:",
"\"Expand right\" → while num + streak in num_set: streak += 1"
],
code: `def longestConsecutive(nums):
numSet = set(nums)
longest = 0
for num in numSet:
if (num - 1) not in numSet:
length = 1
while (num + length) in numSet:
length += 1
longest = max(length, longest)
return longest`,
timeComplexity: "O(n) - we visit each number at most twice",
spaceComplexity: "O(n) - for the set"
}
];

const twoPointersFlashcards = [
{
id: 1,
title: "Valid Palindrome",
question: "Check if a given string is a palindrome, considering only alphanumeric characters and ignoring cases.",
hint: "Try using two pointers moving from both ends of the string, skipping non-alphanumeric characters.",
oneLiner: "Use two pointers from both ends, skipping non-alphanumerics and comparing characters.",
simpleExplanation: "We look at one letter from the front and one from the back.\nWe skip anything that's not a letter or number.\nIf everything matches going inward, it's a palindrome!",
mnemonics: [
"\"Skip non-letters\" → if not s[l].isalnum(): l += 1",
"\"Lowercase compare\" → if s[l].lower() != s[r].lower(): return False",
"\"Move pointers\" → l += 1, r -= 1"
],
code: `def isPalindrome(s: str) -> bool:
left, right = 0, len(s) - 1
while left < right:
while left < right and not s[left].isalnum():
left += 1
while left < right and not s[right].isalnum():
right -= 1
if s[left].lower() != s[right].lower():
return False
left += 1
right -= 1
return True`,
timeComplexity: "O(N) - we process each character at most once",
spaceComplexity: "O(1) - only using two pointers regardless of input size"
},
{
id: 2,
title: "Two Sum II - Input Array Is Sorted",
question: "Find two numbers in a sorted array that add up to a target.",
hint: "Since the array is sorted, try using two pointers from both ends and adjust based on the sum.",
oneLiner: "Use two pointers from both ends and move based on sum.",
simpleExplanation: "We start from both ends of the list.\nIf the sum is too big, move the right one left.\nIf too small, move the left one right — until it fits!",
mnemonics: [
"\"Check sum\" → if nums[l] + nums[r] == target: return",
"\"Too big?\" → if sum > target: r -= 1",
"\"Too small?\" → if sum < target: l += 1"
],
code: `def twoSum(numbers: list[int], target: int) -> list[int]:
left, right = 0, len(numbers) - 1
while left < right:
curr_sum = numbers[left] + numbers[right]
if curr_sum == target:
return [left + 1, right + 1] # 1-based index
if curr_sum < target:
left += 1
else:
right -= 1
return []`,
timeComplexity: "O(N) - we process each element at most once",
spaceComplexity: "O(1) - only using two pointers regardless of input size"
},
{
id: 3,
title: "3Sum",
question: "Find all unique triplets in an array that sum to zero.",
hint: "Sort the array first, then use a combination of iteration and two pointers technique.",
oneLiner: "Sort the array, then for each element, use two pointers to find pairs that complete the triplet.",
simpleExplanation: "First, we sort all numbers.\nThen for each number, we look for two other numbers that add up to create zero.\nWe use two pointers to efficiently find these pairs.",
mnemonics: [
"\"Sort first\" → nums.sort()",
"\"Skip duplicates\" → if i > 0 and nums[i] == nums[i-1]: continue",
"\"Two pointers for sum\" → left, right = i+1, len(nums)-1"
],
code: `def threeSum(nums: list[int]) -> list[list[int]]:
nums.sort()
res = []
for i in range(len(nums) - 2):
if i > 0 and nums[i] == nums[i - 1]: # Skip duplicates
continue
left, right = i + 1, len(nums) - 1
while left < right:
total = nums[i] + nums[left] + nums[right]
if total == 0:
res.append([nums[i], nums[left], nums[right]])
while left < right and nums[left] == nums[left + 1]: # Skip duplicates
left += 1
while left < right and nums[right] == nums[right - 1]: # Skip duplicates
right -= 1
left += 1
right -= 1
elif total < 0:
left += 1
else:
right -= 1
return res`,
timeComplexity: "O(N²) - sorting takes O(N log N), then we have nested loops",
spaceComplexity: "O(1) - excluding the output array"
},
{
id: 4,
title: "Container With Most Water",
question: "Find two vertical lines that form a container with the maximum water storage.",
hint: "Try starting with the widest container and strategically move inward.",
oneLiner: "Use two pointers from ends and move the one with shorter height.",
simpleExplanation: "We look at the widest container first.\nWe always keep the bigger height and try to make width smaller.\nWe keep track of the best result as we move inward.",
mnemonics: [
"\"Start ends\" → l, r = 0, len(height) - 1",
"\"Calculate area\" → area = min(height[l], height[r]) * (r - l)",
"\"Move pointer\" → if height[l] < height[r]: l += 1 else: r -= 1"
],
code: `def maxArea(height: list[int]) -> int:
left, right = 0, len(height) - 1
max_water = 0
while left < right:
max_water = max(max_water, (right - left) * min(height[left], height[right]))
if height[left] < height[right]:
left += 1
else:
right -= 1
return max_water`,
timeComplexity: "O(N) - we process each element at most once",
spaceComplexity: "O(1) - only using constant extra space"
},
{
id: 5,
title: "Trapping Rain Water",
question: "Calculate the amount of rainwater that can be trapped between an array of heights.",
hint: "For each position, the water trapped depends on the maximum heights to its left and right.",
oneLiner: "Use two pointers with left/right max to accumulate water at each position.",
simpleExplanation: "We look at the left and right sides of each block.\nWe keep track of the highest walls on both ends.\nWater is trapped if the block is lower than both sides.",
mnemonics: [
"\"Track maxes\" → left_max = max(left_max, height[l])",
"\"Trap water\" → if height[l] < height[r]: water += left_max - height[l]",
"\"Move inward\" → l += 1 or r -= 1 depending on height"
],
code: `def trap(height: list[int]) -> int:
if not height:
return 0
left, right = 0, len(height) - 1
left_max, right_max = 0, 0
trapped_water = 0
while left < right:
if height[left] < height[right]:
if height[left] >= left_max:
left_max = height[left]
else:
trapped_water += left_max - height[left]
left += 1
else:
if height[right] >= right_max:
right_max = height[right]
else:
trapped_water += right_max - height[right]
right -= 1
return trapped_water`,
timeComplexity: "O(N) - we process each element at most once",
spaceComplexity: "O(1) - using constant extra space"
}
];

const stackFlashcards = [
{
id: 1,
title: "Valid Parentheses",
question: "Given a string containing only ()[], determine if it is valid.",
hint: "Use a stack to match opening and closing brackets.",
oneLiner: "Use a stack to match every closing bracket with the latest opening one.",
simpleExplanation: "Open brackets go into a bag.\nIf you see a closer, match it with the last opener.\nIf the bag is empty at the end, it's valid!",
mnemonics: [
"\"Push opener\" → stack.append(char)",
"\"Pop and match\" → top = stack.pop() if stack else '#'",
"\"Check match\" → if mapping[char] != top: return False"
],
code: `def isValid(s: str) -> bool:
stack = []
mapping = {')': '(', '}': '{', ']': '['}
for char in s:
if char in mapping:
top = stack.pop() if stack else '#'
if mapping[char] != top:
return False
else:
stack.append(char)
return not stack`,
timeComplexity: "O(N) - one pass through the string",
spaceComplexity: "O(N) - in worst case, all opening brackets"
},
{
id: 2,
title: "Min Stack",
question: "Implement a stack that supports push, pop, top, and retrieving the minimum element in constant time.",
hint: "Use an auxiliary stack to track minimum values.",
oneLiner: "Keep a second stack that always holds the current minimum.",
simpleExplanation: "One stack is for real items.\nAnother stack is for the smallest so far.\nPeek the small stack to get the min anytime!",
mnemonics: [
"\"Push min\" → if not min_stack or val <=min_stack[-1]: min_stack.append(val)", "\"Pop min too\" → if popped==min_stack[-1]: min_stack.pop()", "\"Get min\" → return min_stack[-1]" ], code: `class MinStack: def __init__(self): self.stack=[] self.min_stack=[] def push(self, val: int) -> None:
self.stack.append(val)
if not self.min_stack or val <=self.min_stack[-1]: self.min_stack.append(val) def pop(self) -> None:
if self.stack.pop() == self.min_stack[-1]:
self.min_stack.pop()
def top(self) -> int:
return self.stack[-1]
def getMin(self) -> int:
return self.min_stack[-1]`,
timeComplexity: "O(1) - constant time for all operations",
spaceComplexity: "O(N) - in worst case, we store each element twice"
},
{
id: 3,
title: "Evaluate Reverse Polish Notation",
question: "Evaluate an arithmetic expression in Reverse Polish Notation.",
hint: "Use a stack to process numbers and operators.",
oneLiner: "Use a stack to apply operators to previous numbers.",
simpleExplanation: "Read numbers and put them in a stack.\nWhen you see a symbol, do math on the top two numbers.\nPut the result back and keep going!",
mnemonics: [
"\"Pop two, compute\" → a = stack.pop(); b = stack.pop()",
"\"Divide carefully\" → stack.append(int(a / b))",
"\"Push number\" → stack.append(int(token))"
],
code: `def evalRPN(tokens: list[str]) -> int:
stack = []
for token in tokens:
if token in {'+', '-', '*', '/'}:
b, a = stack.pop(), stack.pop()
if token == '+':
stack.append(a + b)
elif token == '-':
stack.append(a - b)
elif token == '*':
stack.append(a * b)
elif token == '/':
stack.append(int(a / b)) # Ensure truncation towards zero
else:
stack.append(int(token))
return stack[0]`,
timeComplexity: "O(N) - process each token once",
spaceComplexity: "O(N) - in worst case, all tokens are numbers"
},
{
id: 4,
title: "Generate Parentheses",
question: "Generate all valid combinations of n pairs of parentheses.",
hint: "Use backtracking with open and close counts.",
oneLiner: "Use backtracking with open and close counters to build valid combinations.",
simpleExplanation: "You can only close if you've opened one.\nKeep adding ( or ) if it's allowed.\nAdd to the list when it's full and balanced!",
mnemonics: [
"\"Base case full\" → if len(s) == 2 * n: res.append(s)",
"\"Try opening\" → if left < n: backtrack(s + \"(\", ...)",
"\"Try closing\" → if right < left: backtrack(s + \")\", ...)"
],
code: `def generateParenthesis(n: int) -> list[str]:
res = []
def backtrack(s, left, right):
if len(s) == 2 * n:
res.append(s)
return
if left < n:
backtrack(s + "(", left + 1, right)
if right < left:
backtrack(s + ")", left, right + 1)
backtrack("", 0, 0)
return res`,
timeComplexity: "O(4^n / √n) - approximate number of valid combinations",
spaceComplexity: "O(N) - max recursion depth of 2n"
},
{
id: 5,
title: "Daily Temperatures",
question: "Given an array of temperatures, return an array where ans[i] is the number of days until a warmer temperature.",
hint: "Use a monotonic stack to track indices of temperatures.",
oneLiner: "Use a monotonic stack to find the next warmer day.",
simpleExplanation: "Keep days in a stack until a warmer one shows up.\nThen pop and mark how long they waited.\nRepeat until all are checked.",
mnemonics: [
"\"Pop when warmer\" → while stack and temperatures[stack[-1]] < temp:",
"\"Set wait days\" → res[idx] = i - idx",
"\"Track index\" → stack.append(i)"
],
code: `def dailyTemperatures(temperatures: list[int]) -> list[int]:
stack = []
res = [0] * len(temperatures)
for i, temp in enumerate(temperatures):
while stack and temperatures[stack[-1]] < temp:
idx = stack.pop()
res[idx] = i - idx
stack.append(i)
return res`,
timeComplexity: "O(N) - each temperature is pushed and popped at most once",
spaceComplexity: "O(N) - in worst case, the stack stores all indices"
},
{
id: 6,
title: "Car Fleet",
question: "Given position and speed arrays of cars, return the number of fleets that arrive at the destination.",
hint: "Sort by position and calculate arrival time for each car.",
oneLiner: "Sort cars by position and use a stack to track merging fleets.",
simpleExplanation: "Sort cars from back to front.\nSee how long each takes to reach the end.\nIf one catches another, they become a team!",
mnemonics: [
"\"Sort by position\" → cars = sorted(zip(position, speed), reverse=True)",
"\"Calc time to end\" → time = (target - pos) / spd",
"\"Stack only new fleets\" → if not stack or time > stack[-1]: stack.append(time)"
],
code: `def carFleet(target: int, position: list[int], speed: list[int]) -> int:
cars = sorted(zip(position, speed), reverse=True)
stack = []
for pos, spd in cars:
time = (target - pos) / spd
if not stack or time > stack[-1]:
stack.append(time)
return len(stack)`,
timeComplexity: "O(N log N) - dominated by the sorting step",
spaceComplexity: "O(N) - for the sorted array and stack"
},
{
id: 7,
title: "Largest Rectangle in Histogram",
question: "Find the largest rectangular area in a histogram.",
hint: "Use a monotonic stack to calculate areas when heights decrease.",
oneLiner: "Use a monotonic stack to compute max area rectangle at each drop.",
simpleExplanation: "Go bar by bar, stack up when taller.\nIf a lower one comes, pop and measure.\nAlways track the biggest area!",
mnemonics: [
"\"Push height index\" → stack.append(i)",
"\"Pop and calc area\" → width = i if not stack else i - stack[-1] - 1",
"\"Update max area\" → max_area = max(max_area, height * width)"
],
code: `def largestRectangleArea(heights: list[int]) -> int:
stack = []
max_area = 0
heights.append(0) # Sentinel value
for i, h in enumerate(heights):
while stack and heights[stack[-1]] > h:
height = heights[stack.pop()]
width = i if not stack else i - stack[-1] - 1
max_area = max(max_area, height * width)
stack.append(i)
return max_area`,
timeComplexity: "O(N) - each bar is pushed and popped at most once",
spaceComplexity: "O(N) - in worst case, the stack contains all bars"
}
];

const binarySearchFlashcards = [
{
id: 1,
title: "Binary Search",
question: "Find the position of a target value in a sorted array.",
hint: "Use the sorted property to cut search space in half each time.",
oneLiner: "Use two pointers to repeatedly cut the search range in half.",
simpleExplanation: "We look in the middle of the list.\nIf it's not the number, we search only the left or right part.\nWe do this again and again until we find it or run out.",
mnemonics: [
"\"Set range\" → low, high = 0, len(nums) - 1",
"\"Check mid\" → mid = (low + high) // 2",
"\"Narrow search\" → if nums[mid] < target: low = mid + 1 else: high = mid - 1"
],
code: `def binarySearch(nums, target):
left, right = 0, len(nums) - 1
while left <=right: mid=(left + right) // 2 if nums[mid]==target: return mid elif nums[mid] < target: left=mid + 1 else: right=mid - 1 return -1`, timeComplexity: "O(log n) - search space halves each iteration", spaceComplexity: "O(1) - constant extra space" }, { id: 2, title: "Search a 2D Matrix", question: "Search for a value in a row and column sorted matrix.", hint: "Treat the 2D matrix as a flattened sorted array for binary search.", oneLiner: "Treat the 2D matrix like a 1D array and use binary search.", simpleExplanation: "We pretend the grid is just one long list.\nWe use binary search on that list.\nWe change the middle number back to row and column to check it.", mnemonics: [ "\"Convert index\" → row=mid // cols; col=mid % cols", "\"Compare mid\" → if matrix[row][col]==target: return True", "\"Adjust range\" → low=mid + 1 or high=mid - 1" ], code: `def searchMatrix(matrix, target): if not matrix or not matrix[0]: return False rows, cols=len(matrix), len(matrix[0]) left, right=0, rows * cols - 1 while left <=right: mid=(left + right) // 2 mid_value=matrix[mid // cols][mid % cols] if mid_value==target: return True elif mid_value < target: left=mid + 1 else: right=mid - 1 return False`, timeComplexity: "O(log(m*n)) - binary search on flattened matrix", spaceComplexity: "O(1) - constant extra space" }, { id: 3, title: "Koko Eating Bananas", question: "Find the minimum eating speed to finish all bananas within a given time.", hint: "Use binary search on the range of possible eating speeds.", oneLiner: "Binary search the eating speed to find the minimum speed that finishes on time.", simpleExplanation: "We try different eating speeds.\nIf she can finish in time, we try slower ones.\nIf not, we try faster ones.", mnemonics: [ "\"Search range\" → low, high=1, max(piles)", "\"Check time\" → total_time=sum(ceil(pile / mid))", "\"Narrow search\" → if time <=h: try slower; else: go faster" ], code: `import math def minEatingSpeed(piles, h): left, right=1, max(piles) def canFinish(speed): return sum(math.ceil(pile / speed) for pile in piles) <=h while left < right: mid=(left + right) // 2 if canFinish(mid): right=mid else: left=mid + 1 return left`, timeComplexity: "O(n log m) - where m is max pile size", spaceComplexity: "O(1) - constant extra space" }, { id: 4, title: "Find Minimum in Rotated Sorted Array", question: "Find the minimum element in a rotated sorted array.", hint: "Use binary search and compare with the rightmost element to determine which half to search.", oneLiner: "Use binary search to find the smallest element by comparing with the rightmost.", simpleExplanation: "The smallest number is in the rotated part.\nWe keep checking the middle and comparing to the end.\nWe move left or right depending on which side is sorted.", mnemonics: [ "\"Mid vs right\" → if nums[mid]> nums[right]: search right",
"\"Else search left\" → right = mid",
"\"Result is nums[left]\" → return nums[left]"
],
code: `def findMin(nums):
left, right = 0, len(nums) - 1
while left < right:
mid = (left + right) // 2
if nums[mid] > nums[right]:
left = mid + 1
else:
right = mid
return nums[left]`,
timeComplexity: "O(log n) - binary search",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 5,
title: "Search in Rotated Sorted Array",
question: "Search for a target in a rotated sorted array.",
hint: "Determine which half is sorted and check if target is in that half.",
oneLiner: "Binary search while checking which half is sorted and target range.",
simpleExplanation: "We split the list in half each time.\nWe check which half is in order.\nThen we pick the side where the target can be.",
mnemonics: [
"\"Left sorted?\" → if nums[low] <=nums[mid]:", "\"Target in left?\" → if nums[low] <=target < nums[mid]: high=mid - 1", "\"Otherwise search right\" → low=mid + 1" ], code: `def search(nums, target): left, right=0, len(nums) - 1 while left <=right: mid=(left + right) // 2 if nums[mid]==target: return mid if nums[left] <=nums[mid]: if nums[left] <=target < nums[mid]: right=mid - 1 else: left=mid + 1 else: if nums[mid] < target <=nums[right]: left=mid + 1 else: right=mid - 1 return -1`, timeComplexity: "O(log n) - binary search", spaceComplexity: "O(1) - constant extra space" }, { id: 6, title: "Time-Based Key-Value Store", question: "Design a time-based key-value store with set and get operations.", hint: "Store values with timestamps in sorted order and use binary search for lookups.", oneLiner: "Store timestamped values in a list and binary search for the latest one <=target time.", simpleExplanation: "We save every version of a value with the time it was saved.\nWhen asked for a time, we look for the latest version at or before that time.\nWe do this using binary search.", mnemonics: [ "\"Store as list\" → store[key].append((timestamp, value))", "\"Binary search timestamp\" → while low <=high: check mid time", "\"Return latest <=timestamp\" → res=value if time <=target" ], code: `from collections import defaultdict import bisect class TimeMap: def __init__(self): self.store=defaultdict(list) def set(self, key, value, timestamp): self.store[key].append((timestamp, value)) def get(self, key, timestamp): if key not in self.store: return "" values=self.store[key] idx=bisect.bisect_right(values, (timestamp, chr(255))) - 1 return values[idx][1] if idx>= 0 else ""`,
timeComplexity: "set: O(1), get: O(log n)",
spaceComplexity: "O(n) - to store all values"
},
{
id: 7,
title: "Median of Two Sorted Arrays",
question: "Find the median of two sorted arrays.",
hint: "Use binary search on the smaller array to find the correct partition point.",
oneLiner: "Use binary search to partition both arrays such that left half ≤ right half.",
simpleExplanation: "We cut both arrays in half in a smart way.\nWe want everything on the left half to be smaller than the right.\nWhen it's balanced, we take the middle numbers for the median.",
mnemonics: [
"\"Binary on smaller array\" → if len(A) > len(B): swap",
"\"Partition and check\" → Aleft <=Bright and Bleft <=Aright", "\"Median calc\" → if total even: avg of max(left), min(right); else: max(left)" ], code: `def findMedianSortedArrays(nums1, nums2): if len(nums1)> len(nums2):
nums1, nums2 = nums2, nums1 # Ensure nums1 is the smaller array
x, y = len(nums1), len(nums2)
left, right = 0, x
while left <=right: partitionX=(left + right) // 2 partitionY=(x + y + 1) // 2 - partitionX maxX=float('-inf') if partitionX==0 else nums1[partitionX - 1] minX=float('inf') if partitionX==x else nums1[partitionX] maxY=float('-inf') if partitionY==0 else nums2[partitionY - 1] minY=float('inf') if partitionY==y else nums2[partitionY] if maxX <=minY and maxY <=minX: if (x + y) % 2==0: return (max(maxX, maxY) + min(minX, minY)) / 2 else: return max(maxX, maxY) elif maxX> minY:
right = partitionX - 1
else:
left = partitionX + 1`,
timeComplexity: "O(log min(m,n)) - binary search on smaller array",
spaceComplexity: "O(1) - constant extra space"
}
];

const slidingWindowFlashcards = [
{
id: 1,
title: "Best Time to Buy and Sell Stock",
question: "Given an array where prices[i] is the price of a stock on day i, find the maximum profit.",
hint: "Track the minimum price seen so far and calculate potential profit at each step.",
oneLiner: "Track the minimum price so far and compute profit at each step.",
simpleExplanation: "We watch the lowest price we've seen.\nThen we check how much we'd earn if we sold today.\nWe remember the best profit we ever found.",
mnemonics: [
"\"Track min\" → min_price = min(min_price, price)",
"\"Check profit\" → profit = price - min_price",
"\"Update best\" → max_profit = max(max_profit, profit)"
],
code: `def maxProfit(prices: list[int]) -> int:
min_price = float('inf')
max_profit = 0
for price in prices:
min_price = min(min_price, price)
max_profit = max(max_profit, price - min_price)
return max_profit`,
timeComplexity: "O(N) - single pass through the array",
spaceComplexity: "O(1) - using constant extra space"
},
{
id: 2,
title: "Longest Substring Without Repeating Characters",
question: "Find the length of the longest substring without repeating characters.",
hint: "Use a sliding window and keep track of characters you've seen.",
oneLiner: "Use a sliding window and a set/map to track characters.",
simpleExplanation: "We move across the string, adding each new letter.\nIf we see a repeat, we shrink the start of the window.\nWe keep track of the longest clean stretch.",
mnemonics: [
"\"Check char in map\" → if s[r] in seen: l = max(l, seen[s[r]] + 1)",
"\"Store index\" → seen[s[r]] = r",
"\"Track max length\" → res = max(res, r - l + 1)"
],
code: `def lengthOfLongestSubstring(s: str) -> int:
char_set = set()
left = max_length = 0
for right in range(len(s)):
while s[right] in char_set:
char_set.remove(s[left])
left += 1
char_set.add(s[right])
max_length = max(max_length, right - left + 1)
return max_length`,
timeComplexity: "O(N) - at most 2n operations in the worst case",
spaceComplexity: "O(min(N, 26)) - limited by character set size"
},
{
id: 3,
title: "Longest Repeating Character Replacement",
question: "Find the length of the longest substring with at most k character replacements.",
hint: "Use a sliding window and track the frequency of each character.",
oneLiner: "Use sliding window; replace excess characters beyond the most frequent one.",
simpleExplanation: "We build a window and count letters inside.\nWe make sure we don't replace more than k characters.\nIf we do, we shrink the window.",
mnemonics: [
"\"Track max freq\" → max_count = max(counts.values())",
"\"Valid window?\" → if (right - left + 1) - max_count > k: shrink",
"\"Update max\" → res = max(res, window length)"
],
code: `from collections import Counter
def characterReplacement(s: str, k: int) -> int:
count = Counter()
left = max_length = max_freq = 0
for right in range(len(s)):
count[s[right]] += 1
max_freq = max(max_freq, count[s[right]])
if (right - left + 1) - max_freq > k:
count[s[left]] -= 1
left += 1
max_length = max(max_length, right - left + 1)
return max_length`,
timeComplexity: "O(N) - single pass through the string",
spaceComplexity: "O(26) - limited to 26 uppercase letters"
},
{
id: 4,
title: "Permutation in String",
question: "Given two strings s1 and s2, check if s1's permutation is a substring of s2.",
hint: "Use a sliding window of length s1 and compare character counts.",
oneLiner: "Use sliding window with frequency counters and compare with target.",
simpleExplanation: "We count letters in the small word.\nThen we move a window over the big word and check each group of letters.\nIf one window matches, we found a match!",
mnemonics: [
"\"Build counters\" → Counter(s1) == Counter(window)",
"\"Slide window\" → add s2[i], remove s2[i - len(s1)]",
"\"Compare counters\" → if match: return True"
],
code: `from collections import Counter
def checkInclusion(s1: str, s2: str) -> bool:
if len(s1) > len(s2):
return False
s1_count = Counter(s1)
s2_count = Counter(s2[:len(s1)])
if s1_count == s2_count:
return True
left = 0
for right in range(len(s1), len(s2)):
s2_count[s2[right]] += 1
s2_count[s2[left]] -= 1
if s2_count[s2[left]] == 0:
del s2_count[s2[left]]
left += 1
if s1_count == s2_count:
return True
return False`,
timeComplexity: "O(N) - where N is the length of s2",
spaceComplexity: "O(26) - limited to 26 lowercase letters"
},
{
id: 5,
title: "Minimum Window Substring",
question: "Given two strings s and t, find the minimum substring of s that contains all characters of t.",
hint: "Use a sliding window and track character frequencies.",
oneLiner: "Use sliding window and hashmap to shrink window once all chars are matched.",
simpleExplanation: "We count the letters we need.\nAs we move forward, we check if we have enough of each letter.\nThen we shrink the window from the left to find the smallest one.",
mnemonics: [
"\"Need map\" → need = Counter(t)",
"\"Expand window\" → window[s[r]] += 1",
"\"Shrink if valid\" → while formed == required: update res, move left"
],
code: `from collections import Counter
def minWindow(s: str, t: str) -> str:
if not t or not s:
return ""
t_count = Counter(t)
window_count = Counter()
left = right = formed = 0
required = len(t_count)
min_len = float("inf")
min_window = ""
while right < len(s):
window_count[s[right]] += 1
if window_count[s[right]] == t_count[s[right]]:
formed += 1
while formed == required:
if right - left + 1 < min_len:
min_len = right - left + 1
min_window = s[left:right+1]
window_count[s[left]] -= 1
if s[left] in t_count and window_count[s[left]] < t_count[s[left]]:
formed -= 1
left += 1
right += 1
return min_window`,
timeComplexity: "O(N) - where N is the length of s",
spaceComplexity: "O(26) - limited to character set size"
},
{
id: 6,
title: "Sliding Window Maximum",
question: "Given an array nums and a window size k, return the maximum element in each window.",
hint: "Use a deque to maintain indices of potential maximum elements.",
oneLiner: "Use a deque to keep track of max values in the current window.",
simpleExplanation: "We go through numbers with a window of size k.\nWe keep only the biggest numbers inside the window.\nWe throw out numbers that are too old or too small.",
mnemonics: [
"\"Pop smaller\" → while q and nums[i] > nums[q[-1]]: q.pop()",
"\"Pop out-of-window\" → if q[0] <=i - k: q.popleft()", "\"Append max\" → if i>= k - 1: res.append(nums[q[0]])"
],
code: `from collections import deque
def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
if not nums:
return []
deque_window = deque()
result = []
for i in range(len(nums)):
# Remove elements outside the window
if deque_window and deque_window[0] < i - k + 1:
deque_window.popleft()
# Remove smaller elements in k range
while deque_window and nums[deque_window[-1]] < nums[i]:
deque_window.pop()
deque_window.append(i)
# Append max value once window size is reached
if i >= k - 1:
result.append(nums[deque_window[0]])
return result`,
timeComplexity: "O(N) - each element is processed at most twice",
spaceComplexity: "O(K) - the deque never grows beyond size k"
}
];

const linkedListFlashcards = [
{
id: 1,
title: "Reverse Linked List",
question: "Reverse a singly linked list.",
hint: "Use three pointers to keep track of previous, current, and next nodes.",
oneLiner: "Iteratively reverse pointers one node at a time.",
simpleExplanation: "We go through the list one by one.\nWe turn each pointer backward instead of forward.\nWhen we finish, the list is flipped!",
mnemonics: [
"\"Track previous\" → prev = None",
"\"Flip pointer\" → curr.next = prev",
"\"Move forward\" → prev, curr = curr, curr.next"
],
code: `def reverseList(head):
prev, curr = None, head
while curr:
nxt = curr.next
curr.next = prev
prev = curr
curr = nxt
return prev`,
timeComplexity: "O(n) - we visit each node once",
spaceComplexity: "O(1) - we use constant extra space"
},
{
id: 2,
title: "Merge Two Sorted Lists",
question: "Merge two sorted linked lists into one sorted list.",
hint: "Use a dummy node to build the merged list.",
oneLiner: "Use two pointers to weave nodes into a sorted list.",
simpleExplanation: "We compare heads of both lists.\nWe always pick the smaller one and keep going.\nWe attach leftover nodes at the end.",
mnemonics: [
"\"Dummy node start\" → dummy = ListNode()",
"\"Pick smaller\" → if l1.val < l2.val: attach l1",
"\"Link remainder\" → tail.next = l1 or l2"
],
code: `def mergeTwoLists(l1, l2):
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
tail.next = l1 if l1 else l2
return dummy.next`,
timeComplexity: "O(n+m) - where n,m are lengths of input lists",
spaceComplexity: "O(1) - we use constant extra space"
},
{
id: 3,
title: "Linked List Cycle",
question: "Detect if a linked list has a cycle.",
hint: "Use fast and slow pointers (Floyd's Cycle Detection).",
oneLiner: "Use two pointers (slow and fast) to detect a loop.",
simpleExplanation: "One pointer moves fast, the other moves slow.\nIf they ever meet, there's a loop.\nIf the fast one finishes, there's no cycle.",
mnemonics: [
"\"Initialize pointers\" → slow, fast = head, head",
"\"Move fast x2\" → fast = fast.next.next",
"\"Check meeting point\" → if slow == fast: return True"
],
code: `def hasCycle(head):
slow, fast = head, head
while fast and fast.next:
slow = slow.next
fast = fast.next.next
if slow == fast:
return True
return False`,
timeComplexity: "O(n) - in worst case, we visit each node once",
spaceComplexity: "O(1) - we use constant extra space"
},
{
id: 4,
title: "Reorder List",
question: "Reorder a linked list in-place as L0→Ln→L1→Ln-1→L2→Ln-2...",
hint: "Find the middle, reverse the second half, and merge the two halves.",
oneLiner: "Split the list, reverse the second half, and merge both.",
simpleExplanation: "We find the middle of the list.\nWe reverse the second half.\nThen we zig-zag merge the two parts.",
mnemonics: [
"\"Find mid\" → slow = slow.next, fast = fast.next.next",
"\"Reverse half\" → second = reverse(middle)",
"\"Merge halves\" → while first and second: alternate attach"
],
code: `def reorderList(head):
if not head or not head.next:
return
# Find middle
slow, fast = head, head
while fast and fast.next:
slow = slow.next
fast = fast.next.next
# Reverse second half
prev, curr = None, slow.next
slow.next = None # Split list
while curr:
nxt = curr.next
curr.next = prev
prev = curr
curr = nxt
# Merge two halves
first, second = head, prev
while second:
tmp1, tmp2 = first.next, second.next
first.next = second
second.next = tmp1
first, second = tmp1, tmp2`,
timeComplexity: "O(n) - we visit each node a constant number of times",
spaceComplexity: "O(1) - we use constant extra space"
},
{
id: 5,
title: "Remove Nth Node From End of List",
question: "Remove the nth node from the end of the list.",
hint: "Use two pointers with a gap of n nodes between them.",
oneLiner: "Use two pointers with n distance apart.",
simpleExplanation: "We move one pointer n steps ahead.\nThen we move both together until the fast one ends.\nThe slow one is right before the node to remove.",
mnemonics: [
"\"Advance fast\" → for _ in range(n): fast = fast.next",
"\"Move both\" → while fast.next: slow = slow.next",
"\"Remove node\" → slow.next = slow.next.next"
],
code: `def removeNthFromEnd(head, n):
dummy = ListNode(0, head)
fast, slow = dummy, dummy
for _ in range(n + 1):
fast = fast.next
while fast:
fast = fast.next
slow = slow.next
slow.next = slow.next.next
return dummy.next`,
timeComplexity: "O(n) - we visit each node at most once",
spaceComplexity: "O(1) - we use constant extra space"
},
{
id: 6,
title: "Copy List with Random Pointer",
question: "Clone a linked list with next and random pointers.",
hint: "Create interleaved list, then fix random pointers, then separate lists.",
oneLiner: "Interleave original and copied nodes, fix randoms, then split.",
simpleExplanation: "We copy each node and put it right next to the original.\nThen we fix the random pointers.\nLast, we split them into two separate lists.",
mnemonics: [
"\"Clone nodes\" → curr.next = Node(curr.val)",
"\"Fix random\" → curr.next.random = curr.random.next",
"\"Separate lists\" → original.next = clone.next; clone.next = clone.next.next"
],
code: `def copyRandomList(head):
if not head:
return None
# Step 1: Create new nodes interleaved with the old nodes
curr = head
while curr:
nxt = curr.next
curr.next = Node(curr.val, nxt, None)
curr = nxt
# Step 2: Assign random pointers
curr = head
while curr:
if curr.random:
curr.next.random = curr.random.next
curr = curr.next.next
# Step 3: Separate the two lists
old, new = head, head.next
new_head = head.next
while old:
old.next = old.next.next if old.next else None
new.next = new.next.next if new.next else None
old = old.next
new = new.next
return new_head`,
timeComplexity: "O(n) - we make three passes through the list",
spaceComplexity: "O(1) - excluding output list"
},
{
id: 7,
title: "Add Two Numbers",
question: "Add two numbers represented as linked lists (digits in reverse order).",
hint: "Track carry and create new nodes as you sum digits.",
oneLiner: "Add digits from each list node by node with carry.",
simpleExplanation: "We add the numbers digit by digit.\nIf the sum is too big, we carry to the next one.\nWe build a new list as we go.",
mnemonics: [
"\"Add values + carry\" → total = l1.val + l2.val + carry",
"\"Carry forward\" → carry = total // 10",
"\"Create node\" → current.next = ListNode(total % 10)"
],
code: `def addTwoNumbers(l1, l2):
dummy = ListNode()
curr, carry = dummy, 0
while l1 or l2 or carry:
val1 = l1.val if l1 else 0
val2 = l2.val if l2 else 0
carry, sum_val = divmod(val1 + val2 + carry, 10)
curr.next = ListNode(sum_val)
curr = curr.next
l1 = l1.next if l1 else None
l2 = l2.next if l2 else None
return dummy.next`,
timeComplexity: "O(max(n,m)) - where n,m are the lengths of the lists",
spaceComplexity: "O(max(n,m)) - for the result list"
},
{
id: 8,
title: "Find The Duplicate Number",
question: "Find the duplicate number in an array where each integer appears only once except for one.",
hint: "Treat array values as pointers and use cycle detection.",
oneLiner: "Use Floyd's Cycle Detection (like Linked List Cycle) on index mapping.",
simpleExplanation: "We pretend the numbers are pointers in a list.\nWe find a cycle using slow and fast.\nThen we find where the cycle begins — that's the duplicate.",
mnemonics: [
"\"Find meeting\" → slow = nums[slow];

const treesFlashcards = [
{
id: 1,
title: "Invert Binary Tree",
question: "Invert a binary tree (swap all left and right children at every node).",
hint: "Apply the inversion recursively at each node.",
oneLiner: "Swap left and right at every node recursively.",
simpleExplanation: "Flip every branch left-to-right.\nDo the same for children too.\nKeep flipping till the bottom!",
mnemonics: [
"\"Flip subtrees\" → root.left, root.right = invertTree(root.right), invertTree(root.left)",
"\"Base case\" → if not root: return None",
"\"Return flipped root\" → return root"
],
code: `def invertTree(self, root):
if not root:
return None
root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
return root`,
timeComplexity: "O(N) - we visit each node once",
spaceComplexity: "O(H) - recursion stack, where H is the height (worst case O(N))"
},
{
id: 2,
title: "Maximum Depth of Binary Tree",
question: "Find the maximum depth (number of nodes along the longest path from root to leaf).",
hint: "Use recursion to find the depth of both subtrees.",
oneLiner: "Recursively get max depth from left and right, add one.",
simpleExplanation: "Go down both left and right.\nFind the deeper one.\nAdd yourself to the count!",
mnemonics: [
"\"Check both sides\" → maxDepth(root.left), maxDepth(root.right)",
"\"Add 1 for current\" → 1 + max(...)",
"\"Stop at null\" → if not root: return 0"
],
code: `def maxDepth(self, root):
if not root:
return 0
return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))`,
timeComplexity: "O(N) - we visit each node once",
spaceComplexity: "O(H) - recursion stack, where H is the height"
},
{
id: 3,
title: "Diameter of Binary Tree",
question: "Find the length of the longest path between any two nodes in a binary tree.",
hint: "Use depth-first search while tracking the longest path through each node.",
oneLiner: "Track the longest path across any node via DFS.",
simpleExplanation: "Find longest path between any two nodes.\nAt every node, try left + right path.\nUpdate the best as you go.",
mnemonics: [
"\"DFS returns depth\" → return 1 + max(left, right)",
"\"Update diameter\" → diameter = max(diameter, left + right)",
"\"Use nonlocal\" → nonlocal diameter"
],
code: `def diameterOfBinaryTree(self, root):
diameter = 0
def depth(node):
nonlocal diameter
if not node:
return 0
left = depth(node.left)
right = depth(node.right)
diameter = max(diameter, left + right)
return 1 + max(left, right)
depth(root)
return diameter`,
timeComplexity: "O(N) - we visit each node once",
spaceComplexity: "O(H) - recursion stack, where H is the height"
},
{
id: 4,
title: "Balanced Binary Tree",
question: "Determine if a binary tree is height-balanced (depth of subtrees differs by at most 1).",
hint: "Check balance while computing height to avoid redundant traversals.",
oneLiner: "Use post-order DFS and return -1 if imbalance is detected.",
simpleExplanation: "Check if both sides are even.\nIf one side too tall, mark broken.\nKeep bubbling -1 up the tree!",
mnemonics: [
"\"Check balance\" → if abs(left - right) > 1: return -1",
"\"Bubble imbalance\" → if left == -1 or right == -1: return -1",
"\"DFS returns height\" → return 1 + max(left, right)"
],
code: `def isBalanced(self, root):
def dfs(node):
if not node:
return 0
left = dfs(node.left)
right = dfs(node.right)
if left == -1 or right == -1 or abs(left - right) > 1:
return -1
return 1 + max(left, right)
return dfs(root) != -1`,
timeComplexity: "O(N) - we visit each node once",
spaceComplexity: "O(H) - recursion stack, where H is the height"
},
{
id: 5,
title: "Same Tree",
question: "Check if two binary trees are the same (have the same structure and values).",
hint: "Compare the nodes recursively from the root down.",
oneLiner: "DFS both trees and match all nodes and values.",
simpleExplanation: "Walk both trees together.\nAt every node, check values.\nIf anything differs, stop!",
mnemonics: [
"\"Match value\" → p.val == q.val",
"\"Match left/right\" → return isSameTree(p.left, q.left) and ...",
"\"Null base case\" → if not p and not q: return True"
],
code: `def isSameTree(self, p, q):
if not p and not q:
return True
if not p or not q or p.val != q.val:
return False
return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)`,
timeComplexity: "O(N) - we visit each node once",
spaceComplexity: "O(H) - recursion stack, where H is the height"
},
{
id: 6,
title: "Subtree of Another Tree",
question: "Check if a tree is a subtree of another tree.",
hint: "Check if current roots match, or if the subtree matches anywhere deeper.",
oneLiner: "Check current node or its children for subtree match.",
simpleExplanation: "Is it the same tree right now?\nIf not, check left and right.\nOne match is all you need!",
mnemonics: [
"\"Check match\" → isSameTree(root, subRoot)",
"\"Recurse left/right\" → isSubtree(root.left, ...) or ...",
"\"Reuse sameTree logic\" → def isSameTree(...)"
],
code: `def isSubtree(self, root, subRoot):
def isSameTree(p, q):
if not p and not q:
return True
if not p or not q or p.val != q.val:
return False
return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
if not root:
return False
return isSameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)`,
timeComplexity: "O(M*N) - where M,N are the sizes of the trees",
spaceComplexity: "O(H) - recursion stack, where H is the height"
},
{
id: 7,
title: "Lowest Common Ancestor of a Binary Search Tree",
question: "Find the lowest common ancestor of two nodes in a binary search tree.",
hint: "Use BST properties to navigate efficiently (values less than root go left, greater go right).",
oneLiner: "Walk down and stop where values split.",
simpleExplanation: "Look left if both are smaller.\nGo right if both are bigger.\nElse, you found the split!",
mnemonics: [
"\"Both left?\" → if p.val < root.val and q.val < root.val:",
"\"Both right?\" → elif p.val > root.val and q.val > root.val:",
"\"Split found\" → return root"
],
code: `def lowestCommonAncestor(self, root, p, q):
while root:
if p.val < root.val and q.val < root.val:
root = root.left
elif p.val > root.val and q.val > root.val:
root = root.right
else:
return root`,
timeComplexity: "O(H) - where H is the height of the tree",
spaceComplexity: "O(1) - iterative approach uses constant space"
},
//backtrackingFlashcards,
export const backtrackingFlashcards = [
{
id: 1,
title: "Subsets",
question: "Generate all possible subsets of a given set of distinct integers.",
hint: "Use backtracking to recursively build all combinations.",
oneLiner: "Use backtracking to include or exclude each number.",
simpleExplanation: "We pick numbers one by one.\nEach time, we decide to keep it or skip it.\nThat gives us all possible combinations.",
mnemonics: [
"\"Start with empty list\" → res = [[]]",
"\"Try adding\" → dfs(index + 1, path + [nums[i]])",
"\"Try skipping\" → dfs(index + 1, path)"
],
code: `def subsets(self, nums):
result = []
def backtrack(start, path):
result.append(path)
for i in range(start, len(nums)):
backtrack(i + 1, path + [nums[i]])
backtrack(0, [])
return result`,
timeComplexity: "O(2^n) - we generate all possible subsets",
spaceComplexity: "O(n) - maximum depth of recursion stack"
},
{
id: 2,
title: "Combination Sum",
question: "Find all unique combinations of candidates where the chosen numbers sum to a target.",
hint: "Use backtracking to try different combinations, and each number can be used unlimited times.",
oneLiner: "Use DFS to try combinations, reusing the same number.",
simpleExplanation: "We keep adding numbers until we hit the target.\nWe can use the same number again.\nIf it's too much, we stop and go back.",
mnemonics: [
"\"Base case\" → if target == 0: add path",
"\"Too big\" → if target < 0: return",
"\"Try again\" → dfs(i, path + [candidates[i]], target - candidates[i])"
],
code: `def combinationSum(self, candidates, target):
result = []
def backtrack(start, target, path):
if target == 0:
result.append(path)
return
for i in range(start, len(candidates)):
if candidates[i] > target:
continue
backtrack(i, target - candidates[i], path + [candidates[i]])
candidates.sort()
backtrack(0, target, [])
return result`,
timeComplexity: "O(N^(T/M)) - where T is target and M is minimum value",
spaceComplexity: "O(T/M) - maximum recursion depth"
},
{
id: 3,
title: "Combination Sum II",
question: "Find all unique combinations in candidates where the chosen numbers sum to target (no duplicates allowed).",
hint: "Sort the array first to handle duplicates easily.",
oneLiner: "Like Combination Sum but skip duplicates and don't reuse elements.",
simpleExplanation: "We can't use the same number again.\nWe also skip duplicates to avoid repeat combos.\nWe keep trying until the sum matches the target.",
mnemonics: [
"\"Sort first\" → candidates.sort()",
"\"Skip duplicates\" → if i > start and candidates[i] == candidates[i - 1]: continue",
"\"DFS without reuse\" → dfs(i + 1, path + [candidates[i]], target - candidates[i])"
],
code: `def combinationSum2(self, candidates, target):
result = []
candidates.sort()
def backtrack(start, target, path):
if target == 0:
result.append(path)
return
prev = -1
for i in range(start, len(candidates)):
if candidates[i] == prev:
continue
if candidates[i] > target:
break
backtrack(i + 1, target - candidates[i], path + [candidates[i]])
prev = candidates[i]
backtrack(0, target, [])
return result`,
timeComplexity: "O(2^n) - worst case, all combinations",
spaceComplexity: "O(n) - maximum recursion depth"
},
{
id: 4,
title: "Permutations",
question: "Return all possible permutations of a list of distinct integers.",
hint: "Track which numbers have been used in the current permutation.",
oneLiner: "Use backtracking to build all possible orders.",
simpleExplanation: "We make all different orders of the numbers.\nWe pick one at a time and don't repeat it.\nWhen the list is full, we save it.",
mnemonics: [
"\"Track used\" → if num in used: continue",
"\"Choose number\" → path.append(num)",
"\"Backtrack\" → path.pop() and used.remove(num)"
],
code: `def permute(self, nums):
result = []
def backtrack(path, options):
if not options:
result.append(path)
return
for i in range(len(options)):
backtrack(path + [options[i]], options[:i] + options[i+1:])
backtrack([], nums)
return result`,
timeComplexity: "O(n!) - we generate all permutations",
spaceComplexity: "O(n) - maximum recursion depth"
},
{
id: 5,
title: "Subsets II",
question: "Return all possible subsets of a list of integers that might contain duplicates.",
hint: "Sort the array and skip duplicates at the same level of recursion.",
oneLiner: "Backtracking with duplicate skip using sorted input.",
simpleExplanation: "Like normal subsets, but now we skip duplicates.\nWe sort the list so we can spot repeats.\nWe only use the first time a number shows up at a level.",
mnemonics: [
"\"Sort input\" → nums.sort()",
"\"Skip repeat at level\" → if i > start and nums[i] == nums[i - 1]: continue",
"\"DFS normally\" → dfs(i + 1, path + [nums[i]])"
],
code: `def subsetsWithDup(self, nums):
result = []
nums.sort()
def backtrack(start, path):
result.append(path)
for i in range(start, len(nums)):
if i > start and nums[i] == nums[i - 1]:
continue
backtrack(i + 1, path + [nums[i]])
backtrack(0, [])
return result`,
timeComplexity: "O(2^n) - worst case, all subsets",
spaceComplexity: "O(n) - maximum recursion depth"
},
{
id: 6,
title: "Word Search",
question: "Given a grid and a word, find if the word exists in the grid. Letters must be adjacent.",
hint: "Use backtracking to explore all possible paths through the grid.",
oneLiner: "Backtrack through board cells matching word letters.",
simpleExplanation: "We try to find each letter one by one.\nWe can only move up/down/left/right.\nIf it works, we return true!",
mnemonics: [
"\"Check match\" → if board[i][j] != word[pos]: return",
"\"Mark visited\" → board[i][j] = '#'",
"\"Unmark (backtrack)\" → board[i][j] = original_letter"
],
code: `def exist(self, board, word):
rows, cols = len(board), len(board[0])
def backtrack(r, c, index):
if index == len(word):
return True
if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[index]:
return False
temp, board[r][c] = board[r][c], '#'
found = (backtrack(r+1, c, index+1) or
backtrack(r-1, c, index+1) or
backtrack(r, c+1, index+1) or
backtrack(r, c-1, index+1))
board[r][c] = temp
return found
for r in range(rows):
for c in range(cols):
if backtrack(r, c, 0):
return True
return False`,
timeComplexity: "O(M×N×4^L) - where L is word length",
spaceComplexity: "O(L) - recursion depth equals word length"
},
{
id: 7,
title: "Palindrome Partitioning",
question: "Partition a string such that every substring is a palindrome. Return all possible partitions.",
hint: "Use backtracking to try different cutting points in the string.",
oneLiner: "Use backtracking to cut the string at every palindrome point.",
simpleExplanation: "We break the string into pieces.\nOnly cut when the piece is a palindrome.\nWe keep cutting until the whole string is split.",
mnemonics: [
"\"Check palindrome\" → if s[l:r+1] == s[l:r+1][::-1]",
"\"Cut and continue\" → dfs(end + 1, path + [s[start:end+1]])",
"\"Add to result\" → if start == len(s): res.append(path)"
],
code: `def partition(self, s):
result = []
def is_palindrome(sub):
return sub == sub[::-1]
def backtrack(start, path):
if start == len(s):
result.append(path)
return
for end in range(start + 1, len(s) + 1):
if is_palindrome(s[start:end]):
backtrack(end, path + [s[start:end]])
backtrack(0, [])
return result`,
timeComplexity: "O(N×2^N) - checking palindromes adds a factor of N",
spaceComplexity: "O(N) - recursion depth is at most N"
},
{
id: 8,
title: "Letter Combinations of a Phone Number",
question: "Given a string containing digits from 2-9, return all possible letter combinations.",
hint: "Create a mapping of digits to letters and use backtracking.",
oneLiner: "Backtrack all digit-letter mappings like a tree.",
simpleExplanation: "Each number maps to letters.\nWe pick one letter per number.\nWe try every combination possible.",
mnemonics: [
"\"Map digits\" → digit_map = {'2': 'abc', ...}",
"\"For each digit\" → for letter in digit_map[digit]:",
"\"Backtrack\" → dfs(index + 1, path + letter)"
],
code: `def letterCombinations(self, digits):
if not digits:
return []
phone = {
'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
'6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
}
result = []
def backtrack(index, path):
if index == len(digits):
result.append(path)
return
for char in phone[digits[index]]:
backtrack(index + 1, path + char)
backtrack(0, '')
return result`,
timeComplexity: "O(4^N) - each digit maps to at most 4 letters",
spaceComplexity: "O(N) - recursion depth equals input length"
},
{
id: 9,
title: "N Queens",
question: "Place N queens on an N×N chessboard so that no two queens attack each other.",
hint: "Track which columns, diagonals, and anti-diagonals are occupied.",
oneLiner: "Place queens row by row, checking for safe spots.",
simpleExplanation: "We try to put queens one row at a time.\nWe make sure they don't attack each other.\nIf it works, we save the board setup.",
mnemonics: [
"\"Loop columns\" → for col in range(n):",
"\"Check safe\" → if col not in cols and row+col not in diag1 and row-col not in diag2",
"\"Place & recurse\" → dfs(row + 1) and backtrack positions"
],
code: `def solveNQueens(self, n):
cols = set()
diag1 = set() # r + c
diag2 = set() # r - c
result = []
board = [['.' for _ in range(n)] for _ in range(n)]
def backtrack(r):
if r == n:
copy = [''.join(row) for row in board]
result.append(copy)
return
for c in range(n):
if c in cols or (r + c) in diag1 or (r - c) in diag2:
continue
cols.add(c)
diag1.add(r + c)
diag2.add(r - c)
board[r][c] = 'Q'
backtrack(r + 1)
cols.remove(c)
diag1.remove(r + c)
diag2.remove(r - c)
board[r][c] = '.'
backtrack(0)
return result`,
timeComplexity: "O(N!) - we try each valid permutation of queen placements",
spaceComplexity: "O(N) - for the board and tracking structures"
}
];

const heapPriorityQueueFlashcards = [
{
id: 1,
title: "Kth Largest Element in a Stream",
question: "Design a class to find the kth largest element in a stream of numbers.",
hint: "Maintain a min-heap of size k to efficiently track the kth largest element.",
oneLiner: "Maintain a min-heap of size `k` to track the Kth largest element.",
simpleExplanation: "We keep the biggest `k` numbers in a bucket.\nWe throw out the smallest one if we get too many.\nThe Kth largest is always the smallest in our bucket.",
mnemonics: [
"\"Push to heap\" → heapq.heappush(self.heap, val)",
"\"Pop if oversized\" → if len(self.heap) > k: heapq.heappop(self.heap)",
"\"Return Kth\" → return self.heap[0]"
],
code: `import heapq
class KthLargest:
def __init__(self, k, nums):
self.k = k
self.min_heap = nums
heapq.heapify(self.min_heap)
while len(self.min_heap) > k:
heapq.heappop(self.min_heap)
def add(self, val):
heapq.heappush(self.min_heap, val)
if len(self.min_heap) > self.k:
heapq.heappop(self.min_heap)
return self.min_heap[0]`,
timeComplexity: "init: O(n log k), add: O(log k)",
spaceComplexity: "O(k) - to store the heap of size k"
},
{
id: 2,
title: "Last Stone Weight",
question: "Smash stones together and return the final weight of the last stone (if any).",
hint: "Use a max-heap (simulated with a min-heap using negative values) to efficiently get the largest stones.",
oneLiner: "Use a max-heap (invert numbers) to always smash the heaviest stones.",
simpleExplanation: "We take the two heaviest stones.\nIf they're different, we smash and put the leftover back.\nWe keep doing that until one or zero stones are left.",
mnemonics: [
"\"Max heap\" → heap = [-x for x in stones]",
"\"Pop two largest\" → a = -heapq.heappop(heap), b = -heapq.heappop(heap)",
"\"Push back if diff\" → if a != b: heapq.heappush(heap, -(a - b))"
],
code: `import heapq
def lastStoneWeight(stones):
stones = [-s for s in stones]
heapq.heapify(stones)
while len(stones) > 1:
first = -heapq.heappop(stones)
second = -heapq.heappop(stones)
if first != second:
heapq.heappush(stones, -(first - second))
return -stones[0] if stones else 0`,
timeComplexity: "O(n log n) - heap operations for each stone",
spaceComplexity: "O(n) - for the heap"
},
{
id: 3,
title: "K Closest Points to Origin",
question: "Find the k closest points to the origin (0, 0) from a list of points.",
hint: "Use a heap to keep track of the k closest points based on their distance from the origin.",
oneLiner: "Use a max-heap of size `k` with negative distances.",
simpleExplanation: "We find how far each point is from the center.\nWe keep only the closest `k` ones.\nAt the end, we give back those points.",
mnemonics: [
"\"Calculate dist\" → dist = x*x + y*y",
"\"Max heap trick\" → heapq.heappush(heap, (-dist, (x, y)))",
"\"Pop if > k\" → if len(heap) > k: heapq.heappop(heap)"
],
code: `import heapq
def kClosest(points, k):
heap = []
for (x, y) in points:
dist = -(x**2 + y**2)
if len(heap) < k:
heapq.heappush(heap, (dist, x, y))
else:
heapq.heappushpop(heap, (dist, x, y))
return [(x, y) for (dist, x, y) in heap]`,
timeComplexity: "O(n log k) - heap operations for each point",
spaceComplexity: "O(k) - for the heap"
},
{
id: 4,
title: "Kth Largest Element in an Array",
question: "Find the kth largest element in an unsorted array.",
hint: "Use a min-heap of size k to track the k largest elements.",
oneLiner: "Use a min-heap of size `k` to track top elements.",
simpleExplanation: "We look through all the numbers.\nWe keep a bucket with the largest `k` ones.\nThe smallest in that bucket is the answer.",
mnemonics: [
"\"Push to heap\" → heapq.heappush(heap, num)",
"\"Trim to k size\" → if len(heap) > k: heapq.heappop(heap)",
"\"Return top\" → return heap[0]"
],
code: `import heapq
def findKthLargest(nums, k):
return heapq.nlargest(k, nums)[-1]
# Alternative implementation:
def findKthLargest_alt(nums, k):
heap = []
for num in nums:
heapq.heappush(heap, num)
if len(heap) > k:
heapq.heappop(heap)
return heap[0]`,
timeComplexity: "O(n log k) - heap operations for each element",
spaceComplexity: "O(k) - for the heap"
},
{
id: 5,
title: "Task Scheduler",
question: "Find the minimum time needed to execute all tasks with cooldown constraints.",
hint: "Greedily schedule the most frequent tasks first to minimize idle time.",
oneLiner: "Use a greedy strategy with a max-heap and cooldown logic.",
simpleExplanation: "We want to do tasks without repeating too soon.\nWe do the most frequent task first.\nIf we wait, we fill gaps with idle time.",
mnemonics: [
"\"Count tasks\" → freq = Counter(tasks)",
"\"Max heap\" → heap = [-cnt for cnt in freq.values()]",
"\"Cooldown cycle\" → for i in range(n + 1): fill tasks or idle"
],
code: `from collections import Counter
import heapq
def leastInterval(tasks, n):
task_counts = Counter(tasks)
max_heap = [-cnt for cnt in task_counts.values()]
heapq.heapify(max_heap)
time = 0
while max_heap:
temp = []
for _ in range(n + 1):
if max_heap:
temp.append(heapq.heappop(max_heap))
for item in temp:
if item + 1 < 0:
heapq.heappush(max_heap, item + 1)
time += n + 1 if max_heap else len(temp)
return time`,
timeComplexity: "O(n) - processing each task once",
spaceComplexity: "O(1) - at most 26 task types (letters)"
},
{
id: 6,
title: "Design Twitter",
question: "Design a simplified version of Twitter with posting, following, and news feed functionality.",
hint: "Use heaps to efficiently merge and retrieve the most recent tweets from followed users.",
oneLiner: "Use a heap to get the 10 most recent tweets across users.",
simpleExplanation: "Each user has their own tweet list.\nWe mix them and keep only the latest 10.\nWe also track who follows whom.",
mnemonics: [
"\"Tweet store\" → self.tweets = {user: []}",
"\"Heap feed\" → heapq.heappush(heap, (-time, tweet))",
"\"Follows set\" → self.followees[user].add(followee)"
],
code: `import heapq
from collections import defaultdict, deque
class Twitter:
def __init__(self):
self.time = 0
self.tweets = defaultdict(deque)
self.followees = defaultdict(set)
def postTweet(self, userId, tweetId):
self.tweets[userId].appendleft((self.time, tweetId))
self.time += 1
def getNewsFeed(self, userId):
heap = []
self.followees[userId].add(userId)
for followee in self.followees[userId]:
for tweet in self.tweets[followee]:
heapq.heappush(heap, tweet)
if len(heap) > 10:
heapq.heappop(heap)
result = []
while heap:
result.append(heapq.heappop(heap)[1])
return result[::-1]
def follow(self, followerId, followeeId):
self.followees[followerId].add(followeeId)
def unfollow(self, followerId, followeeId):
self.followees[followerId].discard(followeeId)`,
timeComplexity: "postTweet: O(1), getNewsFeed: O(F + T log T)",
spaceComplexity: "O(U + T) - users and tweets"
},
{
id: 7,
title: "Find Median from Data Stream",
question: "Design a data structure that supports adding integers and finding the median.",
hint: "Use two heaps to track the lower and upper halves of the data stream.",
oneLiner: "Use two heaps (max-heap + min-heap) to balance lower and upper halves.",
simpleExplanation: "We keep small numbers on one side, big ones on the other.\nThe middle is easy to find when both sides are balanced.\nWe add numbers and move between sides as needed.",
mnemonics: [
"\"Two heaps\" → small = MaxHeap, large = MinHeap",
"\"Balance heaps\" → if len(small) > len(large): move one over",
"\"Get median\" → return (top of small + top of large) / 2"
],
code: `import heapq
class MedianFinder:
def __init__(self):
self.small = [] # Max heap (inverted min heap)
self.large = [] # Min heap
def addNum(self, num):
heapq.heappush(self.small, -num)
if (self.small and self.large and
(-self.small[0] > self.large[0])):
heapq.heappush(self.large, -heapq.heappop(self.small))
if len(self.small) > len(self.large) + 1:
heapq.heappush(self.large, -heapq.heappop(self.small))
if len(self.large) > len(self.small):
heapq.heappush(self.small, -heapq.heappop(self.large))
def findMedian(self):
if len(self.small) > len(self.large):
return -self.small[0]
return (-self.small[0] + self.large[0]) / 2`,
timeComplexity: "addNum: O(log n), findMedian: O(1)",
spaceComplexity: "O(n) - to store all elements"
}
];

const intervalsFlashcards = [
{
id: 1,
title: "Insert Interval",
question: "Given a set of non-overlapping intervals, insert a new interval into the set, merging if necessary.",
hint: "Think about handling intervals before, during, and after the overlap separately.",
oneLiner: "Scan and merge intervals while inserting the new one in the correct place.",
simpleExplanation: "Check which intervals come before the new one.\nThen merge the ones that overlap with it.\nAfter that, just add what's left!",
mnemonics: [
"\"Add before\" → while i < n and intervals[i][1] < newInterval[0]: result.append(intervals[i])",
"\"Merge overlap\" → newInterval[0] = min(newInterval[0], intervals[i][0])",
"\"Add remaining\" → while i < n: result.append(intervals[i])"
],
code: `def insert(intervals, newInterval):
result = []
i = 0
n = len(intervals)
# Add all intervals ending before newInterval starts
while i < n and intervals[i][1] < newInterval[0]:
result.append(intervals[i])
i += 1
# Merge overlapping intervals
while i < n and intervals[i][0] <=newInterval[1]: newInterval[0]=min(newInterval[0], intervals[i][0]) newInterval[1]=max(newInterval[1], intervals[i][1]) i +=1 result.append(newInterval) # Add remaining intervals while i < n: result.append(intervals[i]) i +=1 return result`, timeComplexity: "O(n) - we process each interval once", spaceComplexity: "O(n) - for the result list" }, { id: 2, title: "Merge Intervals", question: "Given a collection of intervals, merge all overlapping intervals.", hint: "Sort intervals by start time to make overlaps contiguous.", oneLiner: "Sort intervals and merge overlapping ones into bigger blocks.", simpleExplanation: "Sort your blocks by when they start.\nIf two blocks touch or overlap, combine them.\nKeep adding non-overlapping ones!", mnemonics: [ "\"Sort by start\" → intervals.sort(key=lambda x: x[0])", "\"Merge check\" → if current[0] <=prev[1]: prev[1]=max(prev[1], current[1])", "\"New block\" → else: merged.append(current)" ], code: `def merge(intervals): if not intervals: return [] # Sort intervals based on the start time intervals.sort(key=lambda x: x[0]) merged=[intervals[0]] for current in intervals[1:]: prev=merged[-1] if current[0] <=prev[1]: # Overlapping intervals prev[1]=max(prev[1], current[1]) else: merged.append(current) return merged`, timeComplexity: "O(n log n) - dominated by the sorting step", spaceComplexity: "O(n) - for the merged list" }, { id: 3, title: "Non-Overlapping Intervals", question: "Find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.", hint: "Sort by end time and greedily select non-overlapping intervals.", oneLiner: "Sort by end time and greedily keep non-overlapping intervals.", simpleExplanation: "Pick intervals that end the earliest.\nIf they don't bump into the next one, keep them.\nSubtract the number you kept from total.", mnemonics: [ "\"Sort by end\" → intervals.sort(key=lambda x: x[1])", "\"Skip overlap\" → if intervals[i][0]>= end:",
"\"Remove extras\" → return len(intervals) - count"
],
code: `def eraseOverlapIntervals(intervals):
if not intervals:
return 0
# Sort intervals based on the end time
intervals.sort(key=lambda x: x[1])
end = intervals[0][1]
count = 1
for i in range(1, len(intervals)):
if intervals[i][0] >= end:
count += 1
end = intervals[i][1]
return len(intervals) - count`,
timeComplexity: "O(n log n) - dominated by the sorting step",
spaceComplexity: "O(1) - using constant extra space"
},
{
id: 4,
title: "Meeting Rooms",
question: "Given an array of meeting time intervals, determine if a person could attend all meetings.",
hint: "Check if any meetings overlap after sorting.",
oneLiner: "Sort intervals and check for any overlaps between meetings.",
simpleExplanation: "Line up all your meetings by start time.\nIf one starts before the last one ended, you can't go!\nNo overlaps? You're good to attend all.",
mnemonics: [
"\"Sort by start\" → intervals.sort(key=lambda x: x[0])",
"\"Overlap check\" → if intervals[i][0] < intervals[i - 1][1]: return False",
"\"All good\" → return True"
],
code: `def canAttendMeetings(intervals):
# Sort intervals based on the start time
intervals.sort(key=lambda x: x[0])
for i in range(1, len(intervals)):
if intervals[i][0] < intervals[i - 1][1]:
return False
return True`,
timeComplexity: "O(n log n) - dominated by the sorting step",
spaceComplexity: "O(1) - using constant extra space"
},
{
id: 5,
title: "Meeting Rooms II",
question: "Find the minimum number of conference rooms required for a set of meeting intervals.",
hint: "Use a min heap to track when rooms become available.",
oneLiner: "Use a min-heap to track meeting end times and allocate rooms.",
simpleExplanation: "Use rooms when meetings overlap.\nFree rooms that end before the next starts.\nCount how many rooms you needed.",
mnemonics: [
"\"Sort by start\" → intervals.sort(key=lambda x: x[0])",
"\"Reuse room\" → if min_heap and min_heap[0] <=interval[0]: heapq.heappop(min_heap)", "\"Add end time\" → heapq.heappush(min_heap, interval[1])" ], code: `import heapq def minMeetingRooms(intervals): if not intervals: return 0 # Sort intervals based on the start time intervals.sort(key=lambda x: x[0]) min_heap=[] for interval in intervals: if min_heap and min_heap[0] <=interval[0]: heapq.heappop(min_heap) heapq.heappush(min_heap, interval[1]) return len(min_heap)`, timeComplexity: "O(n log n) - dominated by sorting and heap operations", spaceComplexity: "O(n) - for the heap in worst case" }, { id: 6, title: "Minimum Interval to Include Each Query", question: "Find the size of the smallest interval that includes each query point.", hint: "Process queries in ascending order and use a min heap to track valid intervals.", oneLiner: "Use a heap to track valid intervals for each sorted query.", simpleExplanation: "Sort your intervals and questions.\nFor each question, keep the smallest interval that fits.\nIf none fit, say \"no size!\"", mnemonics: [ "\"Sort by start/query\" → intervals.sort(), queries_sorted=sorted(enumerate(queries), key=lambda x: x[1])", "\"Push valid\" → heapq.heappush(min_heap, (end - start + 1, end))", "\"Pop expired\" → while min_heap and min_heap[0][1] < query: heapq.heappop(min_heap)" ], code: `import heapq def minInterval(intervals, queries): intervals.sort(key=lambda x: x[0]) queries_sorted=sorted(enumerate(queries), key=lambda x: x[1]) result=[-1] * len(queries) min_heap=[] i=0 for index, query in queries_sorted: while i < len(intervals) and intervals[i][0] <=query: start, end=intervals[i] heapq.heappush(min_heap, (end - start + 1, end)) i +=1 while min_heap and min_heap[0][1] < query: heapq.heappop(min_heap) if min_heap: result[index]=min_heap[0][0] return result`, timeComplexity: "O((n + q) log n) - where n is intervals and q is queries", spaceComplexity: "O(n) - for the heap in worst case" } ];

const greedyFlashcards = [ { id: 1, title: "Maximum Subarray", question: "Find the contiguous subarray within a one-dimensional array of numbers that has the largest sum.", hint: "Use Kadane's algorithm to keep track of the best subarray ending at each position.", oneLiner: "Use Kadane's algorithm to track the best subarray sum ending at each index.", simpleExplanation: "Add each number to your score.\nIf it hurts more than helps, start fresh.\nKeep the highest score!", mnemonics: [ "\"Choose max path\" → current_sum=max(num, current_sum + num)", "\"Track highest\" → max_sum=max(max_sum, current_sum)", "\"Start small\" → current_sum=max_sum=nums[0]" ], code: `def maxSubArray(nums): current_sum=max_sum=nums[0] for num in nums[1:]: current_sum=max(num, current_sum + num) max_sum=max(max_sum, current_sum) return max_sum`, timeComplexity: "O(n) - one pass through the array", spaceComplexity: "O(1) - constant extra space" }, { id: 2, title: "Jump Game", question: "Given an array where each element represents your maximum jump length, determine if you can reach the last index.", hint: "Track the furthest index you can reach as you iterate through the array.", oneLiner: "Track the furthest index reachable at every step.", simpleExplanation: "Jump from stone to stone.\nIf you can't reach one, stop.\nGo as far as you can!", mnemonics: [ "\"Too far to jump?\" → if i> max_reach: return False",
"\"Keep extending\" → max_reach = max(max_reach, i + num)",
"\"Goal achieved\" → return True"
],
code: `def canJump(nums):
max_reach = 0
for i, num in enumerate(nums):
if i > max_reach:
return False
max_reach = max(max_reach, i + num)
if max_reach >= len(nums) - 1:
return True
return True`,
timeComplexity: "O(n) - one pass through the array",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 3,
title: "Jump Game II",
question: "Find the minimum number of jumps to reach the last index.",
hint: "Track the current range and update it when you've explored all current possibilities.",
oneLiner: "Count jumps when current range ends and extend to farthest.",
simpleExplanation: "Each jump gets you closer to the goal.\nWait till you run out of steps.\nThen take the next big jump!",
mnemonics: [
"\"Track farthest\" → farthest = max(farthest, i + nums[i])",
"\"Jump when needed\" → if i == current_end: jumps += 1; current_end = farthest",
"\"Stop at end\" → for i in range(len(nums) - 1):"
],
code: `def jump(nums):
jumps = current_end = farthest = 0
for i in range(len(nums) - 1):
farthest = max(farthest, i + nums[i])
if i == current_end:
jumps += 1
current_end = farthest
return jumps`,
timeComplexity: "O(n) - one pass through the array",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 4,
title: "Gas Station",
question: "Find a starting gas station index to complete a circular route if possible.",
hint: "If total gas >= total cost, there must be a solution. Track the current tank to find it.",
oneLiner: "Track total gas vs cost and reset start when running dry.",
simpleExplanation: "Drive your toy car around the track.\nIf you run out, try the next start.\nIf there's more gas than cost, you'll make it!",
mnemonics: [
"\"Reset on empty\" → if current_tank < 0: start_index = i + 1; current_tank = 0",
"\"Check total\" → return start_index if total_tank >= 0 else -1",
"\"Add net gain\" → total_tank += gas[i] - cost[i]"
],
code: `def canCompleteCircuit(gas, cost):
total_tank = current_tank = start_index = 0
for i in range(len(gas)):
total_tank += gas[i] - cost[i]
current_tank += gas[i] - cost[i]
if current_tank < 0:
start_index = i + 1
current_tank = 0
return start_index if total_tank >= 0 else -1`,
timeComplexity: "O(n) - one pass through the arrays",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 5,
title: "Hand of Straights",
question: "Determine if a hand of cards can be rearranged into groups of consecutive cards.",
hint: "Sort the cards and greedily form groups starting with the smallest available card.",
oneLiner: "Use a min-heap to form straight hands from smallest cards.",
simpleExplanation: "Group cards into stairs like 3-4-5.\nStart with the smallest step.\nIf a step is missing, the group falls!",
mnemonics: [
"\"Build stair group\" → for i in range(first, first + groupSize):",
"\"Check step\" → if count[i] == 0: return False",
"\"Pop if done\" → if count[i] == 0 and i == min_heap[0]: heapq.heappop(min_heap)"
],
code: `from collections import Counter
import heapq
def isNStraightHand(hand, groupSize):
if len(hand) % groupSize != 0:
return False
count = Counter(hand)
min_heap = list(count.keys())
heapq.heapify(min_heap)
while min_heap:
first = min_heap[0]
for i in range(first, first + groupSize):
if count[i] == 0:
return False
count[i] -= 1
if count[i] == 0:
if i != min_heap[0]:
return False
heapq.heappop(min_heap)
return True`,
timeComplexity: "O(n log n) - sorting and heap operations",
spaceComplexity: "O(n) - for the counter and heap"
},
{
id: 6,
title: "Merge Triplets to Form Target Triplet",
question: "Determine if it's possible to merge triplets to form a target triplet.",
hint: "Only consider triplets where each element is less than or equal to the target.",
oneLiner: "Pick only triplets ≤ target and track matched positions.",
simpleExplanation: "Each robot part must be same or smaller.\nFind triplets that match one or more parts.\nCollect all 3 to build your robot!",
mnemonics: [
"\"Filter valid\" → if all(triplet[i] <=target[i] for i in range(3)):", "\"Track matches\" → if triplet[i]==target[i]: good.add(i)", "\"Need all parts\" → return len(good)==3" ], code: `def mergeTriplets(triplets, target): good=set() for triplet in triplets: if all(triplet[i] <=target[i] for i in range(3)): for i in range(3): if triplet[i]==target[i]: good.add(i) return len(good)==3`, timeComplexity: "O(n) - one pass through the triplets", spaceComplexity: "O(1) - constant extra space" }, { id: 7, title: "Partition Labels", question: "Partition a string into as many parts as possible so that each letter appears in at most one part.", hint: "Track the last occurrence of each character and use that to determine partition points.", oneLiner: "Use each character's last occurrence to determine cut points.", simpleExplanation: "You want to cut paper without repeating letters.\nWait until the last of all letters in the part.\nThen make a cut!", mnemonics: [ "\"Last seen\" → last={c: i for i, c in enumerate(S)}", "\"Cut when done\" → if i==j: result.append(i - anchor + 1); anchor=i + 1", "\"Extend end\" → j=max(j, last[c])" ], code: `def partitionLabels(S): last={c: i for i, c in enumerate(S)} j=anchor=0 result=[] for i, c in enumerate(S): j=max(j, last[c]) if i==j: result.append(i - anchor + 1) anchor=i + 1 return result`, timeComplexity: "O(n) - two passes through the string", spaceComplexity: "O(1) - constant extra space (at most 26 letters)" }, { id: 8, title: "Valid Parenthesis String", question: "Determine if a string with '(', ')', and '*' characters can form valid parentheses.", hint: "Track the possible range of open parentheses using two variables.", oneLiner: "Track a possible open-parentheses range with '*' as wildcard.", simpleExplanation: "You're stacking cups.\n'*' can be a cup or nothing.\nJust don't let the stack fall!", mnemonics: [ "\"Range tracking\" → low -=1; high +=1 (for *)", "\"Clamp to zero\" → if low < 0: low=0", "\"Early exit\" → if high < 0: return False" ], code: `def checkValidString(s: str) -> bool:
low = high = 0 # Range of possible open parentheses count
for char in s:
if char == '(':
low += 1
high += 1
elif char == ')':
low -= 1
high -= 1
else: # char == '*'
low -= 1 # treat '*' as ')'
high += 1 # treat '*' as '('
# Clamp low to 0 since we can't have negative open brackets
if high < 0:
return False
if low < 0:
low = 0
return low == 0`,
timeComplexity: "O(n) - one pass through the string",
spaceComplexity: "O(1) - constant extra space"
}
];

const mathGeometryFlashcards = [
{
id: 1,
title: "Rotate Image",
question: "Rotate an n×n matrix 90 degrees clockwise in-place.",
hint: "Think about swapping elements in a strategic way.",
oneLiner: "Transpose the matrix and reverse each row.",
simpleExplanation: "We flip the square along the slanty diagonal.\nThen we flip each row like turning a page.\nNow it looks rotated!",
mnemonics: [
"\"Flip + Transpose\" → matrix[:] = list(zip(*matrix[::-1]))",
"\"Turn inside out\" → Transpose first, reverse next",
"\"In-place magic\" → Modify the same matrix"
],
code: `def rotate(self, matrix: List[List[int]]) -> None:
# Reverse the matrix vertically
matrix.reverse()
# Transpose the matrix
for i in range(len(matrix)):
for j in range(i + 1, len(matrix)):
matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]`,
timeComplexity: "O(N²) - we touch each element in the matrix",
spaceComplexity: "O(1) - we rotate in-place without extra space"
},
{
id: 2,
title: "Spiral Matrix",
question: "Return all elements of a matrix in spiral order, starting from the outside and spiraling inward.",
hint: "Traverse the matrix in a spiral pattern: right, down, left, up, and repeat.",
oneLiner: "Peel outer layers one by one in spiral order.",
simpleExplanation: "We grab the top row, then go down the side.\nThen we go backwards along the bottom, then up.\nRepeat this spiral till everything's picked.",
mnemonics: [
"\"Peel & Rotate\" → res += matrix.pop(0); matrix = list(zip(*matrix))[::-1]",
"\"Right-Down-Left-Up\" → Classic spiral movement",
"\"Shrink the box\" → Matrix gets smaller every loop"
],
code: `def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
res = []
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
steps = [len(matrix[0]), len(matrix) - 1]
r, c, d = 0, -1, 0
while steps[d & 1]:
for i in range(steps[d & 1]):
r += directions[d][0]
c += directions[d][1]
res.append(matrix[r][c])
steps[d & 1] -= 1
d += 1
d %= 4
return res`,
timeComplexity: "O(M×N) - we visit each element once",
spaceComplexity: "O(1) - excluding the output array"
},
{
id: 3,
title: "Set Matrix Zeroes",
question: "Given a matrix, if an element is 0, set its entire row and column to 0.",
hint: "Use the first row and column as markers without using extra space.",
oneLiner: "Use first row and column to mark zeros to be set later.",
simpleExplanation: "If something is zero, we remember its row and column.\nWe use the first row and column as sticky notes.\nLater we turn whole rows and columns to zeros.",
mnemonics: [
"\"Mark for zero\" → matrix[i][0] = matrix[0][j] = 0",
"\"First row/col = hint board\"",
"\"Wipe after scan\" → Do changes only after scanning"
],
code: `def setZeroes(self, matrix: List[List[int]]) -> None:
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
matrix[0][c] = 0`,
timeComplexity: "O(M×N) - we scan the matrix twice",
spaceComplexity: "O(1) - we use the matrix itself to track zeros"
},
{
id: 4,
title: "Happy Number",
question: "Determine if a number is 'happy': replace it with the sum of squares of its digits, and repeat until 1 or a cycle.",
hint: "Use a technique to detect cycles, like fast and slow pointers.",
oneLiner: "Loop sum of squares of digits until 1 or repeat.",
simpleExplanation: "We square and add each digit.\nIf we reach 1, it's a happy number!\nIf we loop, it's stuck and unhappy.",
mnemonics: [
"\"Sum of digit squares\" → n = sum(int(c)**2 for c in str(n))",
"\"Loop with memory\" → seen = set()",
"\"Stop at 1 or cycle\" → while n != 1 and n not in seen: ..."
],
code: `def isHappy(self, n: int) -> bool:
slow, fast = n, self.sumOfSquares(n)
power = lam = 1
while slow != fast:
if power == lam:
slow = fast
power *= 2
lam = 0
fast = self.sumOfSquares(fast)
lam += 1
return True if fast == 1 else False
def sumOfSquares(self, n: int) -> int:
output = 0
while n:
digit = n % 10
digit = digit ** 2
output += digit
n = n // 10
return output`,
timeComplexity: "O(log n) - number of digits decreases quickly",
spaceComplexity: "O(1) - using constant extra space with Floyd's cycle detection"
},
{
id: 5,
title: "Plus One",
question: "Given a non-empty array of digits representing a non-negative integer, increment it by one.",
hint: "Handle carry operations from right to left.",
oneLiner: "Add from the back, handle carry, insert if needed.",
simpleExplanation: "We add 1 to the last digit.\nIf it turns into 10, we carry the 1.\nIf it carries all the way, we add a new digit!",
mnemonics: [
"\"Go backward\" → for i in reversed(range(len(digits)))",
"\"Break early if no carry\"",
"\"Insert 1 if overflow\" → digits.insert(0, 1)"
],
code: `def plusOne(self, digits: List[int]) -> List[int]:
one = 1
i = 0
digits.reverse()
while one:
if i < len(digits):
if digits[i] == 9:
digits[i] = 0
else:
digits[i] += 1
one = 0
else:
digits.append(one)
one = 0
i += 1
digits.reverse()
return digits`,
timeComplexity: "O(N) - potentially examine all digits",
spaceComplexity: "O(1) - excluding output array, constant extra space"
},
{
id: 6,
title: "Pow(x, n)",
question: "Implement pow(x, n), which calculates x raised to the power n.",
hint: "Use binary exponentiation for efficiency.",
oneLiner: "Use divide and conquer to reduce power fast.",
simpleExplanation: "Split the problem into two smaller ones.\nUse the result to build the big one back.\nRepeat until tiny and fast.",
mnemonics: [
"\"Halve and square\" → pow(x * x, n // 2)",
"\"Odd n needs extra x\" → x * pow(...) if n % 2 else pow(...)",
"\"Fast power = Binary Exponentiation\""
],
code: `def myPow(self, x: float, n: int) -> float:
if x == 0:
return 0
if n == 0:
return 1
res = 1
power = abs(n)
while power:
if power & 1:
res *= x
x *= x
power >>= 1
return res if n >= 0 else 1 / res`,
timeComplexity: "O(log n) - binary exponentiation",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 7,
title: "Multiply Strings",
question: "Given two non-negative integers represented as strings, multiply them without converting directly to integers.",
hint: "Simulate the traditional multiplication algorithm digit by digit.",
oneLiner: "Simulate grade-school multiplication using arrays.",
simpleExplanation: "We multiply each digit like on paper.\nAdd to the right place using carry.\nAt the end, remove leading zeroes.",
mnemonics: [
"\"Multiply and place\" → res[i + j + 1] += d1 * d2",
"\"Add carry\" → res[i + j] += res[i + j + 1] // 10",
"\"Join digits\" → ''.join(map(str, res)).lstrip('0')"
],
code: `def multiply(self, num1: str, num2: str) -> str:
if "0" in [num1, num2]:
return "0"
res = [0] * (len(num1) + len(num2))
num1, num2 = num1[::-1], num2[::-1]
for i1 in range(len(num1)):
for i2 in range(len(num2)):
digit = int(num1[i1]) * int(num2[i2])
res[i1 + i2] += digit
res[i1 + i2 + 1] += res[i1 + i2] // 10
res[i1 + i2] = res[i1 + i2] % 10
res, beg = res[::-1], 0
while beg < len(res) and res[beg] == 0:
beg += 1
res = map(str, res[beg:])
return "".join(res)`,
timeComplexity: "O(m*n) - where m,n are the lengths of input strings",
spaceComplexity: "O(m+n) - for the result array"
},
{
id: 8,
title: "Detect Squares",
question: "Design a data structure that supports adding points and counting squares that can be formed.",
hint: "Store points by coordinates and check for square formations efficiently.",
oneLiner: "For each new point, check other 3 corners needed to make a square.",
simpleExplanation: "When we add a point, we remember it.\nTo count squares, we try using that point as one corner.\nWe find 3 other points that make a square with it!",
mnemonics: [
"\"Store count of all points\" → self.points = defaultdict(int)",
"\"Same y = square start\" → Loop all points with same y",
"\"Check distance & match\" → count += self.points[...] * ..."
],
code: `class DetectSquares:
def __init__(self):
self.ptsCount = defaultdict(lambda: defaultdict(int))
def add(self, point: List[int]) -> None:
self.ptsCount[point[0]][point[1]] += 1
def count(self, point: List[int]) -> int:
res = 0
x1, y1 = point
for y2 in self.ptsCount[x1]:
side = y2 - y1
if side == 0:
continue
x3, x4 = x1 + side, x1 - side
res += (self.ptsCount[x1][y2] * self.ptsCount[x3][y1] *
self.ptsCount[x3][y2])
res += (self.ptsCount[x1][y2] * self.ptsCount[x4][y1] *
self.ptsCount[x4][y2])
return res`,
timeComplexity: "add: O(1), count: O(N) where N is the number of points sharing an x-coordinate",
spaceComplexity: "O(N) where N is the total number of points"
}
];

const graphsFlashcards = [
{
id: 1,
title: "Number of Islands",
question: "Given a 2D grid map of '1's (land) and '0's (water), count the number of islands.",
hint: "Use DFS or BFS to explore connected land cells.",
oneLiner: "Use DFS to flood-fill land and count how many times it starts.",
simpleExplanation: "See land? Dive in and mark all connected land.\nGo up, down, left, right — turn it to water.\nCount how many dives you did!",
mnemonics: [
"\"Land found → dive!\" → if grid[r][c] == '1': dfs(r, c); count += 1",
"\"Flood fill\" → grid[r][c] = '0'",
"\"Recursive splash\" → dfs(r + 1, c), dfs(r - 1, c), dfs(r, c + 1), dfs(r, c - 1)"
],
code: `def numIslands(grid):
if not grid:
return 0
def dfs(r, c):
if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == '0':
return
grid[r][c] = '0' # Mark as visited
dfs(r + 1, c)
dfs(r - 1, c)
dfs(r, c + 1)
dfs(r, c - 1)
count = 0
for r in range(len(grid)):
for c in range(len(grid[0])):
if grid[r][c] == '1':
dfs(r, c)
count += 1
return count`,
timeComplexity: "O(M × N), where M is the number of rows and N is the number of columns",
spaceComplexity: "O(M × N) in worst case for recursion stack"
},
{
id: 2,
title: "Max Area of Island",
question: "Find the maximum area of an island in a 2D grid.",
hint: "Track area while using DFS/BFS to explore islands.",
oneLiner: "DFS to count connected land areas and track the max size.",
simpleExplanation: "Explore each island like a treasure map.\nCount each land you step on.\nKeep the biggest number!",
mnemonics: [
"\"Step on land\" → return 1 + dfs(...) + ...",
"\"Sink it\" → grid[r][c] = 0",
"\"Track best\" → max_area = max(max_area, dfs(r, c))"
],
code: `def maxAreaOfIsland(grid):
if not grid:
return 0
def dfs(r, c):
if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == 0:
return 0
grid[r][c] = 0 # Mark as visited
return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)
max_area = 0
for r in range(len(grid)):
for c in range(len(grid[0])):
if grid[r][c] == 1:
max_area = max(max_area, dfs(r, c))
return max_area`,
timeComplexity: "O(M × N), where M is rows and N is columns",
spaceComplexity: "O(M × N) in worst case for recursion stack"
},
{
id: 3,
title: "Clone Graph",
question: "Create a deep copy of a connected undirected graph.",
hint: "Use a hash map to map original nodes to their clones.",
oneLiner: "Use DFS and a hashmap to copy each node and its neighbors.",
simpleExplanation: "Copy the node. Then copy its friends.\nDon't copy the same kid twice!\nUse a notebook to remember who you copied.",
mnemonics: [
"\"Memoize node\" → old_to_new[node] = copy",
"\"Visit neighbors\" → copy.neighbors.append(dfs(neighbor))",
"\"Return copy\" → return old_to_new[node]"
],
code: `class Node:
def __init__(self, val=0, neighbors=None):
self.val = val
self.neighbors = neighbors if neighbors is not None else []
def cloneGraph(node):
if not node:
return None
old_to_new = {}
def dfs(node):
if node in old_to_new:
return old_to_new[node]
copy = Node(node.val)
old_to_new[node] = copy
for neighbor in node.neighbors:
copy.neighbors.append(dfs(neighbor))
return copy
return dfs(node)`,
timeComplexity: "O(N + M), where N is nodes and M is edges",
spaceComplexity: "O(N) for mapping old nodes to new nodes"
},
{
id: 4,
title: "Walls and Gates",
question: "Fill each empty room with the distance to its nearest gate.",
hint: "Use multi-source BFS starting from all gates.",
oneLiner: "Use BFS from every gate to fill rooms with the shortest distance.",
simpleExplanation: "Start walking from every open gate.\nStep by step, add 1 to your count.\nStop if you hit a wall!",
mnemonics: [
"\"Start from gates\" → if rooms[r][c] == 0: queue.append((r, c))",
"\"Step update\" → rooms[rr][cc] = rooms[r][c] + 1",
"\"BFS directions\" → for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]"
],
code: `from collections import deque
def wallsAndGates(rooms):
if not rooms:
return
rows, cols = len(rooms), len(rooms[0])
queue = deque()
for r in range(rows):
for c in range(cols):
if rooms[r][c] == 0:
queue.append((r, c))
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
while queue:
r, c = queue.popleft()
for dr, dc in directions:
rr, cc = r + dr, c + dc
if 0 <=rr < rows and 0 <=cc < cols and rooms[rr][cc]==float('inf'): rooms[rr][cc]=rooms[r][c] + 1 queue.append((rr, cc))`, timeComplexity: "O(M × N), where M is rows and N is columns", spaceComplexity: "O(M × N) for the queue in worst case" }, { id: 5, title: "Rotting Oranges", question: "Determine the minimum time required for all oranges to become rotten.", hint: "Use BFS with a time counter to track each minute of rot spread.", oneLiner: "Use BFS to spread rot from all rotten oranges minute by minute.", simpleExplanation: "Rotten oranges spread the stink.\nEach minute, they infect their neighbors.\nIf any fresh ones are left, return -1!", mnemonics: [ "\"Start with all rotten\" → if grid[r][c]==2: queue.append((r, c, 0))", "\"Spread the rot\" → grid[rr][cc]=2", "\"Track time\" → minutes=max(minutes, cur_minute)" ], code: `from collections import deque def orangesRotting(grid): if not grid: return -1 rows, cols=len(grid), len(grid[0]) queue=deque() fresh_oranges=0 for r in range(rows): for c in range(cols): if grid[r][c]==2: queue.append((r, c, 0)) elif grid[r][c]==1: fresh_oranges +=1 directions=[(1, 0), (-1, 0), (0, 1), (0, -1)] minutes=0 while queue: r, c, minutes=queue.popleft() for dr, dc in directions: rr, cc=r + dr, c + dc if 0 <=rr < rows and 0 <=cc < cols and grid[rr][cc]==1: grid[rr][cc]=2 fresh_oranges -=1 queue.append((rr, cc, minutes + 1)) return minutes if fresh_oranges==0 else -1`, timeComplexity: "O(M × N), where M is rows and N is columns", spaceComplexity: "O(M × N) for the queue in worst case" }, { id: 6, title: "Pacific Atlantic Water Flow", question: "Find grid coordinates where water can flow to both Pacific and Atlantic oceans.", hint: "Start from ocean edges and work inward to find reachable cells.", oneLiner: "DFS from both ocean edges to find cells that can reach both.", simpleExplanation: "Water flows downhill or flat.\nStart from each ocean and mark the cells.\nReturn where both oceans meet!", mnemonics: [ "\"Visit cell\" → visited.add((r, c))", "\"DFS only higher/equal\" → if heights[r][c] < prev_height: return", "\"Intersect both oceans\" → return list(pacific & atlantic)" ], code: `def pacificAtlantic(heights): if not heights: return [] m, n=len(heights), len(heights[0]) pacific=set() atlantic=set() def dfs(r, c, visited, prev_height): if ( (r, c) in visited or r < 0 or c < 0 or r>= m or c >= n or
heights[r][c] < prev_height
):
return
visited.add((r, c))
for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
dfs(r + dr, c + dc, visited, heights[r][c])
for i in range(m):
dfs(i, 0, pacific, heights[i][0])
dfs(i, n - 1, atlantic, heights[i][n - 1])
for j in range(n):
dfs(0, j, pacific, heights[0][j])
dfs(m - 1, j, atlantic, heights[m - 1][j])
return list(pacific & atlantic)`,
timeComplexity: "O(m × n), where m is rows and n is columns",
spaceComplexity: "O(m × n) for the sets and recursion stack"
},
{
id: 7,
title: "Surrounded Regions",
question: "Capture all regions surrounded by X in a board.",
hint: "Mark border-connected O's first, then flip the rest.",
oneLiner: "Use DFS to mark safe 'O's on the border, then flip the rest.",
simpleExplanation: "Mark the border O's as safe.\nFlip all trapped O's to X.\nFlip the safe ones back.",
mnemonics: [
"\"Safe marker\" → board[r][c] = '#'",
"\"Flip trapped\" → if board[i][j] == 'O': board[i][j] = 'X'",
"\"Restore safe\" → if board[i][j] == '#': board[i][j] = 'O'"
],
code: `def solve(board):
if not board or not board[0]:
return
m, n = len(board), len(board[0])
def dfs(r, c):
if r < 0 or c < 0 or r >= m or c >= n or board[r][c] != 'O':
return
board[r][c] = '#'
for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
dfs(r + dr, c + dc)
for i in range(m):
dfs(i, 0)
dfs(i, n - 1)
for j in range(n):
dfs(0, j)
dfs(m - 1, j)
for i in range(m):
for j in range(n):
if board[i][j] == 'O':
board[i][j] = 'X'
elif board[i][j] == '#':
board[i][j] = 'O'`,
timeComplexity: "O(m × n), each cell processed at most twice",
spaceComplexity: "O(m × n) for the recursion stack in worst case"
},
{
id: 8,
title: "Course Schedule",
question: "Determine if it's possible to finish all courses given prerequisites.",
hint: "Use topological sort to check if there's any cycle in the graph.",
oneLiner: "Use topological sort to check if all courses can be completed.",
simpleExplanation: "Take courses with no prerequisites.\if len(temp) > len(longest): longest = temp"
],
code: `def longestPalindrome(s):
def expand_around_center(left, right):
while left >= 0 and right < len(s) and s[left] == s[right]:
left -= 1
right += 1
return s[left + 1:right]
longest = ""
for i in range(len(s)):
# Odd length palindrome
temp = expand_around_center(i, i)
if len(temp) > len(longest):
longest = temp
# Even length palindrome
temp = expand_around_center(i, i + 1)
if len(temp) > len(longest):
longest = temp
return longest`,
timeComplexity: "O(n²) - expand up to n times for each center",
spaceComplexity: "O(1) - constant extra space (not counting result)"
},
//oneDDPFlashcards,
export const oneDDPFlashcards = [
{
id: 1,
title: "Climbing Stairs",
question: "Given n steps, you can climb 1 or 2 steps at a time. Find the number of distinct ways to reach the top.",
hint: "Think about how many ways you can reach each step from previous steps.",
oneLiner: "Use Fibonacci-style bottom-up DP to count steps.",
simpleExplanation: "You can climb 1 or 2 stairs at a time.\nCount how many ways to get to each stair.\nIt's like adding ways from two previous steps.",
mnemonics: [
"\"Fibonacci step\" → first, second = second, first + second",
"\"Start base\" → first, second = 1, 2",
"\"Return top\" → return second"
],
code: `def climbStairs(n):
if n <=2: return n first, second=1, 2 for _ in range(3, n + 1): first, second=second, first + second return second`, timeComplexity: "O(n) - one pass through the steps", spaceComplexity: "O(1) - constant extra space" }, { id: 2, title: "Min Cost Climbing Stairs", question: "Given an array cost where cost[i] is the cost of the i-th step, find the minimum cost to reach the top.", hint: "For each step, decide whether to jump from 1 or 2 steps back based on minimum cost.", oneLiner: "Track min cost to reach each stair by choosing the cheaper path.", simpleExplanation: "Every stair has a price.\nYou can jump one or two ahead.\nAlways pay the cheaper way!", mnemonics: [ "\"Take cheaper path\" → first, second=second, cost[i] + min(first, second)", "\"Start base\" → first, second=cost[0], cost[1]", "\"Choose last\" → return min(first, second)" ], code: `def minCostClimbingStairs(cost): n=len(cost) first, second=cost[0], cost[1] for i in range(2, n): first, second=second, cost[i] + min(first, second) return min(first, second)`, timeComplexity: "O(n) - one pass through the cost array", spaceComplexity: "O(1) - constant extra space" }, { id: 3, title: "House Robber", question: "Given an array of house values, find the maximum amount you can rob without robbing adjacent houses.", hint: "For each house, decide whether to rob it (and skip the previous) or skip it.", oneLiner: "Use DP to choose max between robbing current or skipping it.", simpleExplanation: "Can't rob two houses in a row.\nEach time, choose: rob this or skip it.\nKeep track of best steal.", mnemonics: [ "\"Rob or skip\" → first, second=second, max(second, first + num)", "\"Track rolling max\" → first, second=0, 0", "\"Final loot\" → return second" ], code: `def rob(nums): if not nums: return 0 if len(nums)==1: return nums[0] first, second=0, 0 for num in nums: first, second=second, max(second, first + num) return second`, timeComplexity: "O(n) - one pass through nums array", spaceComplexity: "O(1) - constant extra space" }, { id: 4, title: "House Robber II", question: "Similar to House Robber, but houses are in a circle. Find the maximum amount you can rob.", hint: "Break the circle by solving the problem twice, once skipping the first house and once skipping the last.", oneLiner: "Run house robber on both circle-split paths and return the best.", simpleExplanation: "First and last houses are neighbors.\nSo rob from 0 to n-2 or from 1 to n-1.\nTake the better of the two!", mnemonics: [ "\"Rob linearly\" → rob_linear(nums[:-1]), rob_linear(nums[1:])", "\"Rolling max again\" → first, second=second, max(second, first + num)", "\"Return max plan\" → return max(...)" ], code: `def rob(nums): def rob_linear(houses): first, second=0, 0 for num in houses: first, second=second, max(second, first + num) return second if len(nums)==1: return nums[0] return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))`, timeComplexity: "O(n) - two linear passes", spaceComplexity: "O(1) - constant extra space" }, { id: 5, title: "Longest Palindromic Substring", question: "Find the longest palindromic substring in a given string.", hint: "Expand around centers to find palindromes, considering both odd and even length cases.", oneLiner: "Expand around each center to find longest mirror substring.", simpleExplanation: "A palindrome reads the same both ways.\nTry expanding around every letter.\nKeep the longest one you find!", mnemonics: [ "\"Expand center\" → while s[left]==s[right]: left--, right++", "\"Two tries\" → expand(i, i), expand(i, i+1)", "\"Track longest\" → title: "Binary Tree Level Order Traversal", question: "Return the level order traversal of a binary tree (values grouped by levels).", hint: "Use breadth-first search with a queue to process nodes level by level.", oneLiner: "Use a queue to collect node values level by level.", simpleExplanation: "Start at the root.\nVisit every level, left to right.\nAdd children to the queue!", mnemonics: [ "\"Loop level size\" → for _ in range(len(queue)):", "\"Push children\" → if node.left: queue.append(...)", "\"Save level\" → result.append(level)" ], code: `from collections import deque def levelOrder(self, root): if not root: return [] queue, result=deque([root]), [] while queue: level=[] for _ in range(len(queue)): node=queue.popleft() level.append(node.val) if node.left: queue.append(node.left) if node.right: queue.append(node.right) result.append(level) return result`, timeComplexity: "O(N) - we visit each node once", spaceComplexity: "O(N) - in worst case, the queue stores all nodes at the deepest level" }, { id: 9, title: "Binary Tree Right Side View", question: "Return the values visible from the right side of a binary tree (rightmost node at each level).", hint: "Use level order traversal and keep the rightmost node at each level.", oneLiner: "Save the last node of each level from left-to-right scan.", simpleExplanation: "You stand on the right side.\nSee the last node of each row.\nOnly those are visible!", mnemonics: [ "\"Peek last\" → result.append(queue[-1].val)", "\"BFS as usual\" → for _ in range(len(queue)):", "\"Push children\" → queue.append(node.left/right)" ], code: `from collections import deque def rightSideView(self, root): if not root: return [] queue, result=deque([root]), [] while queue: result.append(queue[-1].val) for _ in range(len(queue)): node=queue.popleft() if node.left: queue.append(node.left) if node.right: queue.append(node.right) return result`, timeComplexity: "O(N) - we visit each node once", spaceComplexity: "O(D) - where D is the tree's max width" }, { id: 10, title: "Count Good Nodes in Binary Tree", question: "Count nodes where all ancestors have values <=the node's value.", hint: "Track the maximum value seen on the path from root to current node.", oneLiner: "DFS down the tree and count nodes>= max seen so far.",
simpleExplanation: "Start from root.\nIf you're bigger than all parents, count yourself.\nPass the max as you go down.",
mnemonics: [
"\"Compare with max\" → count = 1 if node.val >= maxVal else 0",
"\"Pass down max\" → maxVal = max(maxVal, node.val)",
"\"Sum results\" → return count + dfs(left) + dfs(right)"
],
code: `def goodNodes(self, root):
def dfs(node, maxVal):
if not node:
return 0
count = 1 if node.val >= maxVal else 0
maxVal = max(maxVal, node.val)
return count + dfs(node.left, maxVal) + dfs(node.right, maxVal)
return dfs(root, root.val)`,
timeComplexity: "O(N) - we visit each node once",
spaceComplexity: "O(H) - recursion stack, where H is the height"
}
];

const trieFlashcards = [
{
id: 1,
title: "Implement Trie (Prefix Tree)",
question: "Design a data structure that supports inserting, searching, and prefix matching for strings",
hint: "Think about how to represent character-by-character traversal using nested dictionaries",
oneLiner: "Use a nested dictionary where each node is a character map, ending with a special end marker.",
simpleExplanation: "We build a tree where each letter has its own branch.\nWe follow each letter step by step when adding a word.\nA special symbol tells us when a word ends.",
mnemonics: [
"\"Start root\" → self.root = {}",
"\"Insert char by char\" → node = node.setdefault(char, {})",
"\"Mark end of word\" → node['#'] = True"
],
code: `class TrieNode:
def __init__(self):
self.children = {}
self.isWord = False
class Trie:
def __init__(self):
self.root = TrieNode()
def insert(self, word):
node = self.root
for ch in word:
if ch not in node.children:
node.children[ch] = TrieNode()
node = node.children[ch]
node.isWord = True
def search(self, word):
node = self._find(word)
return node is not None and node.isWord
def startsWith(self, prefix):
return self._find(prefix) is not None
def _find(self, word):
node = self.root
for ch in word:
if ch not in node.children:
return None
node = node.children[ch]
return node`,
timeComplexity: "insert: O(L), search: O(L), startsWith: O(L) where L is word length",
spaceComplexity: "O(N) where N is total characters inserted"
},
{
id: 2,
title: "Design Add and Search Words Data Structure",
question: "Implement a data structure that can add words and search for words with wildcards",
hint: "How would you handle a '.' character that can match any letter during search?",
oneLiner: "Extend Trie with DFS to handle wildcards (.) during search.",
simpleExplanation: "We store words in a special tree (Trie).\nWhen searching, we can use . to mean any letter.\nWe check all possible paths for matches.",
mnemonics: [
"\"Dot means explore\" → if char == '.': try all children",
"\"End match\" → if at end and '#' in node: return True",
"\"DFS search\" → searchHelper(word, index, node)"
],
code: `class WordDictionary:
def __init__(self):
self.root = {}
def addWord(self, word):
node = self.root
for ch in word:
node = node.setdefault(ch, {})
node['#'] = True # End of word
def search(self, word):
def dfs(node, i):
if i == len(word):
return '#' in node
if word[i] == '.':
return any(dfs(child, i + 1) for child in node if child != '#')
return word[i] in node and dfs(node[word[i]], i + 1)
return dfs(self.root, 0)`,
timeComplexity: "addWord: O(L), search: Worst case O(26^L) if all characters are .",
spaceComplexity: "O(N) where N is total characters inserted"
},
{
id: 3,
title: "Word Search II",
question: "Given an m×n board of characters and a list of words, find all words that can be formed on the board",
hint: "Could you use a Trie to quickly check if a prefix exists in the word list during board traversal?",
oneLiner: "Build a Trie of words, then DFS through board to match prefixes.",
simpleExplanation: "We put all words into a search tree (Trie).\nThen we walk around the board letter by letter.\nIf we match a word path, we add it to our answers.",
mnemonics: [
"\"Build Trie first\" → for word in words: insert(word)",
"\"Explore neighbors\" → dfs(i, j, node)",
"\"Found word\" → if '#' in node: add to result"
],
code: `class TrieNode:
def __init__(self):
self.children = {}
self.word = None # Store word at the end
class Solution:
def findWords(self, board, words):
root = TrieNode()
# Build Trie
for word in words:
node = root
for ch in word:
if ch not in node.children:
node.children[ch] = TrieNode()
node = node.children[ch]
node.word = word
res = []
rows, cols = len(board), len(board[0])
def dfs(r, c, node):
char = board[r][c]
if char not in node.children:
return
nxt_node = node.children[char]
if nxt_node.word:
res.append(nxt_node.word)
nxt_node.word = None # Avoid duplicates
board[r][c] = '#'
for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
nr, nc = r + dr, c + dc
if 0 <=nr < rows and 0 <=nc < cols and board[nr][nc] !='#': dfs(nr, nc, nxt_node) board[r][c]=char for r in range(rows): for c in range(cols): dfs(r, c, root) return res`, timeComplexity: "Building Trie: O(W * L), DFS Search: O(M * N * 4^L)", spaceComplexity: "O(W * L) for Trie + O(L) recursion depth" } ];

const twoDDPFlashcards = [ { id: 1, title: "Unique Paths", question: "Find the number of unique paths from top-left to bottom-right of an m×n grid, moving only right or down.", hint: "Think about how many ways you can reach each cell in the grid.", oneLiner: "Build bottom-up by storing path counts from the bottom row to the top.", simpleExplanation: "You can move right or down in a grid.\nEach cell holds total paths from there.\nStart at the bottom and fill the top!", mnemonics: [ "\"Start with rightmost path as 1s\" → row=[1] * n", "\"Move bottom-up\" → newRow[j]=newRow[j+1] + row[j]", "\"Top-left holds the answer\" → return row[0]" ], code: `def uniquePaths(self, m: int, n: int) -> int:
row = [1] * n
for i in range(m - 1):
newRow = [1] * n
for j in range(n - 2, -1, -1):
newRow[j] = newRow[j + 1] + row[j]
row = newRow
return row[0]`,
timeComplexity: "O(m×n) - we process each cell once",
spaceComplexity: "O(n) - we only store two rows at a time"
},
{
id: 2,
title: "Longest Common Subsequence",
question: "Find the length of the longest subsequence common to two strings.",
hint: "Use a 2D table to track matching characters between the strings.",
oneLiner: "Compare characters bottom-up and store best matches.",
simpleExplanation: "Look for common letters in two words.\nIf letters match, grow your chain.\nRemember best paths from the future!",
mnemonics: [
"\"Match adds 1\" → curr[j] = 1 + prev[j + 1]",
"\"Else take max\" → curr[j] = max(curr[j + 1], prev[j])",
"\"Swap rows\" → prev, curr = curr, prev"
],
code: `def longestCommonSubsequence(self, text1: str, text2: str) -> int:
if len(text1) < len(text2):
text1, text2 = text2, text1
prev = [0] * (len(text2) + 1)
curr = [0] * (len(text2) + 1)
for i in range(len(text1) - 1, -1, -1):
for j in range(len(text2) - 1, -1, -1):
if text1[i] == text2[j]:
curr[j] = 1 + prev[j + 1]
else:
curr[j] = max(curr[j + 1], prev[j])
prev, curr = curr, prev
return prev[0]`,
timeComplexity: "O(m×n) - where m,n are the lengths of the strings",
spaceComplexity: "O(min(m,n)) - we only store two rows"
},
{
id: 3,
title: "Best Time to Buy/Sell Stock III",
question: "Find the maximum profit from at most two transactions (buy one and sell one share).",
hint: "Track the best profit for different states: after first buy, first sell, second buy, and second sell.",
oneLiner: "Track two transactions using rolling DP updates from right to left.",
simpleExplanation: "You can buy and sell twice.\nTrack your best profits in reverse.\nKeep updating what to buy/sell!",
mnemonics: [
"\"Buy max profit\" → dp_buy = max(dp1_sell - price, dp1_buy)",
"\"Sell max profit\" → dp_sell = max(dp2_buy + price, dp1_sell)",
"\"Shift state\" → dp2_buy = dp1_buy"
],
code: `def maxProfit(self, prices: List[int]) -> int:
n = len(prices)
dp1_buy, dp1_sell = 0, 0
dp2_buy = 0
for i in range(n - 1, -1, -1):
dp_buy = max(dp1_sell - prices[i], dp1_buy)
dp_sell = max(dp2_buy + prices[i], dp1_sell)
dp2_buy = dp1_buy
dp1_buy, dp1_sell = dp_buy, dp_sell
return dp1_buy`,
timeComplexity: "O(n) - one pass through the prices array",
spaceComplexity: "O(1) - using constant extra space"
},
{
id: 4,
title: "Coin Change II",
question: "Find the number of combinations that make up an amount using an array of coins.",
hint: "Use a bottom-up approach, considering each coin one at a time.",
oneLiner: "Use DP to count all combinations to reach an amount.",
simpleExplanation: "You can use each coin as many times.\nTry every way to make the target.\nCount how many combos reach it!",
mnemonics: [
"\"Init base case\" → dp[0] = 1",
"\"Use each coin\" → nextDP[a] += nextDP[a - coin]",
"\"Swap arrays\" → dp = nextDP"
],
code: `def change(self, amount: int, coins: List[int]) -> int:
dp = [0] * (amount + 1)
dp[0] = 1
for i in range(len(coins) - 1, -1, -1):
nextDP = [0] * (amount + 1)
nextDP[0] = 1
for a in range(1, amount + 1):
nextDP[a] = dp[a]
if a - coins[i] >= 0:
nextDP[a] += nextDP[a - coins[i]]
dp = nextDP
return dp[amount]`,
timeComplexity: "O(amount × n) - where n is the number of coins",
spaceComplexity: "O(amount) - for the DP arrays"
},
{
id: 5,
title: "Target Sum",
question: "Find ways to assign + and - to each number in an array to reach a target sum.",
hint: "Instead of using a 2D array, use maps to track all possible sums at each step.",
oneLiner: "DP maps every possible sum from ± choices of nums.",
simpleExplanation: "Each number can be + or –.\nTry both for every total.\nCount all paths to your goal!",
mnemonics: [
"\"Try both signs\" → next_dp[total ± num] += count",
"\"Loop through totals\" → for total, count in dp.items():",
"\"Final result\" → return dp[target]"
],
code: `def findTargetSumWays(self, nums: List[int], target: int) -> int:
dp = defaultdict(int)
dp[0] = 1
for num in nums:
next_dp = defaultdict(int)
for total, count in dp.items():
next_dp[total + num] += count
next_dp[total - num] += count
dp = next_dp
return dp[target]`,
timeComplexity: "O(n×sum) - where sum is the sum of all numbers",
spaceComplexity: "O(sum) - for the dictionary"
},
{
id: 6,
title: "Interleaving String",
question: "Determine if s3 is an interleaving of s1 and s2 (preserving order of characters).",
hint: "Use DP to track whether we can form s3 up to a certain point using s1 and s2.",
oneLiner: "Use 1D DP to check if s3 can be formed by interleaving s1 and s2.",
simpleExplanation: "Mix two strings to make the third.\nCheck every possible merge.\nDon't lose track of match order!",
mnemonics: [
"\"Check char match\" → if s1[i] == s3[i+j] and dp[j]",
"\"Also try s2\" → if s2[j] == s3[i+j] and nextDp",
"\"Update DP\" → dp[j] = res"
],
code: `def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
m, n = len(s1), len(s2)
if m + n != len(s3):
return False
if n < m:
s1, s2 = s2, s1
m, n = n, m
dp = [False for _ in range(n + 1)]
dp[n] = True
for i in range(m, -1, -1):
nextDp = True
for j in range(n - 1, -1, -1):
res = False
if i < m and s1[i] == s3[i + j] and dp[j]:
res = True
if j < n and s2[j] == s3[i + j] and nextDp:
res = True
dp[j] = res
nextDp = dp[j]
return dp[0]`,
timeComplexity: "O(m×n) - where m,n are the lengths of s1 and s2",
spaceComplexity: "O(min(m,n)) - for the DP array"
},
{
id: 7,
title: "Longest Increasing Path in Matrix",
question: "Find the length of the longest increasing path in a matrix.",
hint: "Use memoization to avoid recalculating the same cell multiple times.",
oneLiner: "DFS + memoization to get the longest increasing path from each cell.",
simpleExplanation: "Climb from one number to bigger ones.\nRemember best steps from each point.\nReturn the longest trail!",
mnemonics: [
"\"Cache results\" → if (r, c) in dp: return dp[(r, c)]",
"\"Try all 4 dirs\" → dfs(r ± 1, c), dfs(r, c ± 1)",
"\"Max of all paths\" → return max(dp.values())"
],
code: `def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
ROWS, COLS = len(matrix), len(matrix[0])
dp = {} # (r, c) -> LIP
def dfs(r, c, prevVal):
if (r < 0 or r == ROWS or c < 0 or
c == COLS or matrix[r][c] <=prevVal ): return 0 if (r, c) in dp: return dp[(r, c)] res=1 res=max(res, 1 + dfs(r + 1, c, matrix[r][c])) res=max(res, 1 + dfs(r - 1, c, matrix[r][c])) res=max(res, 1 + dfs(r, c + 1, matrix[r][c])) res=max(res, 1 + dfs(r, c - 1, matrix[r][c])) dp[(r, c)]=res return res for r in range(ROWS): for c in range(COLS): dfs(r, c, -1) return max(dp.values())`, timeComplexity: "O(m×n) - each cell is processed once", spaceComplexity: "O(m×n) - for the cache" }, { id: 8, title: "Distinct Subsequences", question: "Count the number of distinct subsequences of s that equal t.", hint: "Use DP to track the number of ways to form t[0...j] using s[0...i].", oneLiner: "Use DP to count all ways s can form t as a subsequence.", simpleExplanation: "You can skip letters in s to form t.\nEvery match gives a new way.\nAdd the number of paths together!", mnemonics: [ "\"Base case\" → dp[n]=1", "\"Match char\" → if s[i]==t[j]: res +=prev", "\"Update dp\" → dp[j]=res" ], code: `def numDistinct(self, s: str, t: str) -> int:
m, n = len(s), len(t)
dp = [0] * (n + 1)
dp[n] = 1
for i in range(m - 1, -1, -1):
prev = 1
for j in range(n - 1, -1, -1):
res = dp[j]
if s[i] == t[j]:
res += prev
prev = dp[j]
dp[j] = res
return dp[0]`,
timeComplexity: "O(m×n) - where m,n are the lengths of s and t",
spaceComplexity: "O(n) - for the DP array"
},
{
id: 9,
title: "Edit Distance",
question: "Find the minimum number of operations to convert one string to another.",
hint: "Use DP to track the minimum operations needed for each prefix of both strings.",
oneLiner: "Bottom-up DP tracks minimum ops to convert one string to another.",
simpleExplanation: "To match two words, you can insert, delete, or replace.\nTake the fewest steps possible.\nTry from the end back to the start.",
mnemonics: [
"\"Match? Skip\" → if word1[i] == word2[j]: dp[j] = nextDp",
"\"Else try all\" → dp[j] = 1 + min(dp[j], dp[j + 1], nextDp)",
"\"Track next\" → nextDp = temp"
],
code: `def minDistance(self, word1: str, word2: str) -> int:
m, n = len(word1), len(word2)
if m < n:
m, n = n, m
word1, word2 = word2, word1
dp = [n - i for i in range(n + 1)]
for i in range(m - 1, -1, -1):
nextDp = dp[n]
dp[n] = m - i
for j in range(n - 1, -1, -1):
temp = dp[j]
if word1[i] == word2[j]:
dp[j] = nextDp
else:
dp[j] = 1 + min(dp[j], dp[j + 1], nextDp)
nextDp = temp
return dp[0]`,
timeComplexity: "O(m×n) - where m,n are the lengths of the strings",
spaceComplexity: "O(min(m,n)) - for the DP array"
},
{
id: 10,
title: "Burst Balloons",
question: "Find the maximum coins you can collect by bursting balloons.",
hint: "Use DP to consider the last balloon to burst in each range.",
oneLiner: "Use interval DP to track max coins from bursting every balloon last in range.",
simpleExplanation: "Pop balloons for coins!\nTry each as the last in range.\nStore the best coins for each section.",
mnemonics: [
"\"Add padding\" → new_nums = [1] + nums + [1]",
"\"Burst last in (l,r)\" → coins = L * i * R + dp[l][i-1] + dp[i+1][r]",
"\"Try all i\" → for i in range(l, r + 1):"
],
code: `def maxCoins(self, nums):
n = len(nums)
new_nums = [1] + nums + [1]
dp = [[0] * (n + 2) for _ in range(n + 2)]
for l in range(n, 0, -1):
for r in range(l, n + 1):
for i in range(l, r + 1):
coins = new_nums[l - 1] * new_nums[i] * new_nums[r + 1]
coins += dp[l][i - 1] + dp[i + 1][r]
dp[l][r] = max(dp[l][r], coins)
return dp[1][n]`,
timeComplexity: "O(n³) - three nested loops",
spaceComplexity: "O(n²) - for the 2D DP array"
},
{
id: 11,
title: "Regular Expression Matching",
question: "Implement regular expression matching with support for '.' and '*'.",
hint: "Use DP to track whether subpatterns match substrings.",
oneLiner: "Match s and p with backtracking and support for . and *.",
simpleExplanation: "Match characters step-by-step.\nIf * shows up, skip or use it.\nUse memory to speed up match!",
mnemonics: [
"\"Check match\" → s[i] == p[j] or p[j] == '.'",
"\"Handle '*'\" → res = dp[j + 2] or (match and dp[j])",
"\"Update DP\" → dp[j], dp1 = res, dp[j]"
],
code: `def isMatch(self, s: str, p: str) -> bool:
dp = [False] * (len(p) + 1)
dp[len(p)] = True
for i in range(len(s), -1, -1):
dp1 = dp[len(p)]
dp[len(p)] = (i == len(s))
for j in range(len(p) - 1, -1, -1):
match = i < len(s) and (s[i] == p[j] or p[j] == ".")
res = False
if (j + 1) < len(p) and p[j + 1] == "*":
res = dp[j + 2]
if match:
res |= dp[j]
elif match:
res = dp1
dp[j], dp1 = res, dp[j]
return dp[0]`,
timeComplexity: "O(m×n) - where m,n are the lengths of s and p",
spaceComplexity: "O(n) - for the DP array"
}
];

const bitManipulationFlashcards = [
{
id: 1,
title: "Single Number",
question: "Find the single number that appears only once in an array where all other numbers appear twice.",
hint: "Consider using the XOR operation to cancel out duplicates.",
oneLiner: "Use XOR to cancel out all duplicate numbers, leaving the single one.",
simpleExplanation: "Every number appears twice except one.\nMatching numbers cancel each other out.\nOnly the lonely one stays!",
mnemonics: [
"\"Cancel duplicates\" → result ^= num",
"\"Start with zero\" → result = 0",
"\"Return the lone survivor\" → return result"
],
code: `def singleNumber(nums):
result = 0
for num in nums:
# XOR operation: a ^ a = 0 and a ^ 0 = a
# So all pairs will cancel out, leaving only the single number
result ^= num
return result`,
timeComplexity: "O(n) - one pass through the array",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 2,
title: "Number of 1 Bits",
question: "Count the number of '1' bits in an unsigned integer.",
hint: "A powerful bit manipulation trick: n & (n-1) clears the rightmost set bit.",
oneLiner: "Use n & (n - 1) to clear the rightmost 1 until n becomes 0.",
simpleExplanation: "Look at a number's binary.\nEvery time you chop off a 1, count it.\nKeep chopping till there are none!",
mnemonics: [
"\"Chop rightmost 1\" → n &= (n - 1)",
"\"Count each chop\" → count += 1",
"\"Loop until empty\" → while n:"
],
code: `def hammingWeight(n):
count = 0
while n:
# n & (n-1) removes the rightmost 1 bit
n &= (n - 1)
count += 1
return count`,
timeComplexity: "O(k) - where k is the number of 1 bits",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 3,
title: "Counting Bits",
question: "Count the number of '1' bits in each number from 0 to n.",
hint: "Reuse previously calculated results with dynamic programming.",
oneLiner: "Build up the 1-bit count using previous results and last bit.",
simpleExplanation: "Every number is made from smaller ones.\nCopy their 1-count and add the last bit.\nStore the count and keep going!",
mnemonics: [
"\"Right shift reuse\" → result[i >> 1] + (i & 1)",
"\"Start from zero\" → result = [0] * (n + 1)",
"\"Build bottom-up\" → for i in range(1, n + 1):"
],
code: `def countBits(n):
# Initialize result array with 0 for the first element
result = [0] * (n + 1)
for i in range(1, n + 1):
# For any number i, result[i] = result[i >> 1] + (i & 1)
# This works because i >> 1 is i / 2, and i & 1 is the last bit
result[i] = result[i >> 1] + (i & 1)
return result`,
timeComplexity: "O(n) - we process each number from 0 to n",
spaceComplexity: "O(n) - for the output array"
},
{
id: 4,
title: "Reverse Bits",
question: "Reverse the bits of a 32-bit unsigned integer.",
hint: "Build the result bit by bit, shifting and OR-ing.",
oneLiner: "Shift and build the reversed number bit-by-bit.",
simpleExplanation: "Read one bit at a time from right to left.\nPut it in the new number from left to right.\nDo this 32 times!",
mnemonics: [
"\"Shift left & add\" → result = (result << 1) | (n & 1)", "\"Shift n right\" → n>>= 1",
"\"Do it 32 times\" → for i in range(32):"
],
code: `def reverseBits(n):
result = 0
for i in range(32):
# Left shift result by 1 to make space for the next bit
result <<=1 # Add the least significant bit of n to result result |=(n & 1) # Right shift n by 1 to process the next bit n>>= 1
return result`,
timeComplexity: "O(1) - constant time (always 32 operations)",
spaceComplexity: "O(1) - constant extra space"
},
{
id: 5,
title: "Missing Number",
question: "Find the missing number in a sequence of 0 to n.",
hint: "Use properties of XOR to find the missing element.",
oneLiner: "Use XOR to cancel out all matching numbers and find the missing one.",
simpleExplanation max_heap = [-cnt for cnt in task_counts.values()]
heapq.heapify(max_heap)
time = 0
while max_heap:
temp = []
for _ in range(n + 1):
if max_heap:
temp.append(heapq.heappop(max_heap))
for item in temp:
if item + 1 < 0:
heapq.heappush(max_heap, item + 1)
time += n + 1 if max_heap else len(temp)
return time`,
timeComplexity: "O(n) - processing each task once",
spaceComplexity: "O(1) - at most 26 task types (letters)"
},
{
id: 6,
title: "Design Twitter",
question: "Design a simplified version of Twitter with posting, following, and news feed functionality.",
hint: "Use heaps to efficiently merge and retrieve the most recent tweets from followed users.",
oneLiner: "Use a heap to get the 10 most recent tweets across users.",
simpleExplanation: "Each user has their own tweet list.\nWe mix them and keep only the latest 10.\nWe also track who follows whom.",
mnemonics: [
"\"Tweet store\" → self.tweets = {user: []}",
"\"Heap feed\" → heapq.heappush(heap, (-time, tweet))",
"\"Follows set\" → self.followees[user].add(followee)"
],
code: `import heapq
from collections import defaultdict, deque
class Twitter:
def __init__(self):
self.time = 0
self.tweets = defaultdict(deque)
self.followees = defaultdict(set)
def postTweet(self, userId, tweetId):
self.tweets[userId].appendleft((self.time, tweetId))
self.time += 1
def getNewsFeed(self, userId):
heap = []
self.followees[userId].add(userId)
for followee in self.followees[userId]:
for tweet in self.tweets[followee]:
heapq.heappush(heap, tweet)
if len(heap) > 10:
heapq.heappop(heap)
result = []
while heap:
result.append(heapq.heappop(heap)[1])
return result[::-1]
def follow(self, followerId, followeeId):
self.followees[followerId].add(followeeId)
def unfollow(self, followerId, followeeId):
self.followees[followerId].discard(followeeId)`,
timeComplexity: "postTweet: O(1), getNewsFeed: O(F + T log T)",
spaceComplexity: "O(U + T) - users and tweets"
},
{
id: 7,
title: "Find Median from Data Stream",
question: "Design a data structure that supports adding integers and finding the median.",
hint: "Use two heaps to track the lower and upper halves of the data stream.",
oneLiner: "Use two heaps (max-heap + min-heap) to balance lower and upper halves.",
simpleExplanation: "We keep small numbers on one side, big ones on the other.\nThe middle is easy to find when both sides are balanced.\nWe add numbers and move between sides as needed.",
mnemonics: [
"\"Two heaps\" → small = MaxHeap, large = MinHeap",
"\"Balance heaps\" → if len(small) > len(large): move one over",
"\"Get median\" → return (top of small + top of large) / 2"
],
code: `import heapq
class MedianFinder:
def __init__(self):
self.small = [] # Max heap (inverted min heap)
self.large = [] # Min heap
def addNum(self, num):
heapq.heappush(self.small, -num)
if (self.small and self.large and
(-self.small[0] > self.large[0])):
heapq.heappush(self.large, -heapq.heappop(self.small))
if len(self.small) > len(self.large) + 1:
heapq.heappush(self.large, -heapq.heappop(self.small))
if len(self.large) > len(self.small):
heapq.heappush(self.small, -heapq.heappop(self.large))
def findMedian(self):
if len(self.small) > len(self.large):
return -self.small[0]
return (-self.small[0] + self.large[0]) / 2`,
timeComplexity: "addNum: O(log n), findMedian: O(1)",
spaceComplexity: "O(n) - to store all elements"
}
];