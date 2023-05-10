## Count the Number of Digits in an Integer
'''
Given a decimal number, continue dividing the number by ten until it reaches 0 and records the remainder at each step.
The resulting list of remainders is the equivalent place values of the Integer.
---
def countDigit(n):
  count=0
  while n>0:
    count+=1
    n=n//10
  return count

n=125
print("Number of digits: ",countDigit(n))

O(n) - O(1)
'''

## Convert Decimal Number to Binary Number
'''
Bit-shifting
def DecimalToBinary(n):
    count=0
    while(n>0):
        count+=1
        n>>=1
    return count
---
Optimal?
def DecimalToBinary(n):
    bitCounter=0
    while((1 << bitCounter) <= n):
        bitCounter+=1
    return bitCounter
'''

### *** AND
## Count Set Bits
'''
Basic  - O(1) O(1)

def helper(n):
    count=0
    while n > 0:
         count+=(n & 1)
         n >>= 1
       
    return count
---
Optimized
# Brian Kernighan’s algorithm
In this approach, we count only the set bits. So,
- If a number has 2 set bits, then the while loop runs two times.
- If a number has 4 set bits, then the while loop runs four times.
Leave the left most bit as is:
  n  = 40           => 00000000 00000000 00000000 00101000
  n - 1 = 39        => 00000000 00000000 00000000 00100111
-----------------------------------------------------------
(n & (n - 1)) = 32  => 00000000 00000000 00000000 00100000   
-----------------------------------------------------------  
def CountSetBits(n):
    count=0
    while n > 0:
        n &= (n - 1)
        count+=1
    return count  
# Lookup table

def CountSetBits(n):
    table = [0] * 256
    table[0] = 0

    for i in range(1, 256):
        table[i] = (i & 1) + table[i >> 1] 

    res = 0
    for i in range(4): 
        res += table[n & 0xff] 
        n >>= 8 

    return res
'''

## Counting Bits II
'''
Write a program to return an array of number of 1’s in the binary representation of every number in the range [0, n].
---
def count_bits(n):
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

def counting_bits(n):
    ans = [0] * (n + 1)
    for i in range(n + 1):
        ans[i] = count_bits(i)
    return ans

n=6
print(counting_bits(n))

# output: Result [0, 1, 1, 2, 1, 2, 2]
---
O(n) - O(1)
'''
