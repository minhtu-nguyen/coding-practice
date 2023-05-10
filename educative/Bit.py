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
## Check If Number Is Even/Odd
'''
It would be best if you saw the pattern of bits present in odd numbers. If you take a closer look at each of them, you can see the right-most significant bit is set to 1 (for 2^0 place).
So, we do an AND bitwise operation with 1 decimal number to see if the resultant output is 1. If not, it is an even number else odd.
---
def IsEven(array):
    result = []

    def helper(array):
        k = 0
        for n in array:
            result.append("Odd" if (n & 1) == 1 else "Even")
            k += 1
        return result

    return helper(array)

print(IsEven([1, 2, 3, 4, 5, 6, 7, 8, 9]))

O(n) - O(n)
'''

## Power of 2
'''
# Brute-force
def IsPowerOf2(n):
    if n == 0:
        return False
    while n != 1:
        if n %2 !=0:
            return False
        n >>= 1
    return True
O(logn) - O(1)
# Brian Kernighan’s algorithm
If a number is the power of 2, we know that only one set bit is present in its Binary representation ~ n & (n - 1)==0
def helper(n):
    if n == 0:
        return False
    return (n & (n - 1) == 0)
# One line
def IsPowerOf2(n):
    return n != 0 and (n & (n - 1)==0)
O(1) - O(1)
'''
### *** OR
## Number Of Flips Required To Make a|b Equal to c
'''
The flip operation consists of changing any single bit 1 to 0 or changing the bit 0 to 1 in their binary representation.
---
def MinFlips(a,b,c):
    ans=0
    for i in range(0,32):
        bitC = ((c >> i) & 1)
        bitA = ((a >> i) & 1)
        bitB = ((b >> i) & 1)

        if((bitA | bitB) != bitC):
            if(bitC == 0):
                if(bitA == 1 and bitB == 1):
                     ans=ans+2
                else:
                    ans=ans+1 # bitA or bitB == 0
            else: # bitC == 1
                        ans=ans+1 # bitA and bitB == 0
    return ans

a=2
b=6
c=5
print("Min Flips required to make two numbers equal to third is : ",MinFlips(a,b,c))
---
O(logn) - O(1)
'''

### *** NOT
## Switch Sign of a Number
'''
  number   = 8
  ~number  = -9
----------------------------
~number + 1 = (-9 + 1) = -8
----------------------------
def switchSign(number):
      return ~number + 1
---
O(1) - O(1)
'''

### *** XOR
## Swap Two Numbers
'''
def swap_nums(a,b):
    a=a^b
    b=b^a
    a=a^b
    print("Finally, after swapping a =", a, ",b =", b)

a=10
b=121
swap_nums(a,b)
---
O(1) - O(1)
'''

## Find Odd Occurring Element
'''
In this question, every element appears an even number of times except for one element, which appears an odd number of times. The element that appears an odd number of times is our answer.
---
If we take XOR of zero and a bit, it will return that bit. a ^ 0 = a
If we take XOR of duplicate bits, it will return 0. a ^ a = 0
For n numbers, the same below math can be applied.
a ^ b ^ a = (a ^ a) ^ b = 0 ^ b = b;
---
def OddOccurence(array):
    res=0
    for value in array:
        res=res^value
    return res

array=[4,3,3,4,4,4,5,3,5]
print("Odd occuring element is : ",str(OddOccurence(array)))
---
O(n) - O(1)
'''

## Detect If Two Integers Have Opposite Signs
'''
The XOR rule says the output will be 1 only when two input values are opposite (0 and 1, or 1 and 0).
The above concept clearly says if the leading left-MSB (left-most significant bit) is 1, then it’s negative.
The following four examples will clearly explain these concepts.
Consider two numbers whose left MSBs are both 1, and the XOR of them is 0.
If two numbers both have left MSBs of 0, then the XOR of them is 0.
Now, when the left MSB of two inputs is different, then the XOR yields to 1, which is a negative number.
So, when we perform (x ^ y) < 0, we get the right answer.
---
def oppositeSigns(x,y):
  return "signs are opposite" if (x ^ y)< 0 else "signs are not opposite"

x=100
y=-1
print("For inputs ",x,"," ,y ,":" ,oppositeSigns(x,y))
---
O(1) - O(1)
'''

##