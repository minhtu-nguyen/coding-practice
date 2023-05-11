### *** Arithmetic Analysis
## Bisection
from collections.abc import Callable


def bisection(function: Callable[[float], float], a: float, b: float) -> float:
    """
    finds where function becomes 0 in [a,b] using bolzano
    >>> bisection(lambda x: x ** 3 - 1, -5, 5)
    1.0000000149011612
    >>> bisection(lambda x: x ** 3 - 1, 2, 1000)
    Traceback (most recent call last):
        ...
    ValueError: could not find root in given interval.
    >>> bisection(lambda x: x ** 2 - 4 * x + 3, 0, 2)
    1.0
    >>> bisection(lambda x: x ** 2 - 4 * x + 3, 2, 4)
    3.0
    >>> bisection(lambda x: x ** 2 - 4 * x + 3, 4, 1000)
    Traceback (most recent call last):
        ...
    ValueError: could not find root in given interval.
    """
    start: float = a
    end: float = b
    if function(a) == 0:  # one of the a or b is a root for the function
        return a
    elif function(b) == 0:
        return b
    elif (
        function(a) * function(b) > 0
    ):  # if none of these are root and they are both positive or negative,
        # then this algorithm can't find the root
        raise ValueError("could not find root in given interval.")
    else:
        mid: float = start + (end - start) / 2.0
        while abs(start - mid) > 10**-7:  # until precisely equals to 10^-7
            if function(mid) == 0:
                return mid
            elif function(mid) * function(start) < 0:
                end = mid
            else:
                start = mid
            mid = start + (end - start) / 2.0
        return mid


def f(x: float) -> float:
    return x**3 - 2 * x - 5


if __name__ == "__main__":
    print(bisection(f, 1, 1000))

    import doctest

    doctest.testmod()

## Gaussian Elimination
"""
Gaussian elimination method for solving a system of linear equations.
Gaussian elimination - https://en.wikipedia.org/wiki/Gaussian_elimination
"""


import numpy as np
from numpy import float64
from numpy.typing import NDArray


def retroactive_resolution(
    coefficients: NDArray[float64], vector: NDArray[float64]
) -> NDArray[float64]:
    """
    This function performs a retroactive linear system resolution
        for triangular matrix

    Examples:
        2x1 + 2x2 - 1x3 = 5         2x1 + 2x2 = -1
        0x1 - 2x2 - 1x3 = -7        0x1 - 2x2 = -1
        0x1 + 0x2 + 5x3 = 15
    >>> gaussian_elimination([[2, 2, -1], [0, -2, -1], [0, 0, 5]], [[5], [-7], [15]])
    array([[2.],
           [2.],
           [3.]])
    >>> gaussian_elimination([[2, 2], [0, -2]], [[-1], [-1]])
    array([[-1. ],
           [ 0.5]])
    """

    rows, columns = np.shape(coefficients)

    x: NDArray[float64] = np.zeros((rows, 1), dtype=float)
    for row in reversed(range(rows)):
        total = 0
        for col in range(row + 1, columns):
            total += coefficients[row, col] * x[col]

        x[row, 0] = (vector[row] - total) / coefficients[row, row]

    return x


def gaussian_elimination(
    coefficients: NDArray[float64], vector: NDArray[float64]
) -> NDArray[float64]:
    """
    This function performs Gaussian elimination method

    Examples:
        1x1 - 4x2 - 2x3 = -2        1x1 + 2x2 = 5
        5x1 + 2x2 - 2x3 = -3        5x1 + 2x2 = 5
        1x1 - 1x2 + 0x3 = 4
    >>> gaussian_elimination([[1, -4, -2], [5, 2, -2], [1, -1, 0]], [[-2], [-3], [4]])
    array([[ 2.3 ],
           [-1.7 ],
           [ 5.55]])
    >>> gaussian_elimination([[1, 2], [5, 2]], [[5], [5]])
    array([[0. ],
           [2.5]])
    """
    # coefficients must to be a square matrix so we need to check first
    rows, columns = np.shape(coefficients)
    if rows != columns:
        return np.array((), dtype=float)

    # augmented matrix
    augmented_mat: NDArray[float64] = np.concatenate((coefficients, vector), axis=1)
    augmented_mat = augmented_mat.astype("float64")

    # scale the matrix leaving it triangular
    for row in range(rows - 1):
        pivot = augmented_mat[row, row]
        for col in range(row + 1, columns):
            factor = augmented_mat[col, row] / pivot
            augmented_mat[col, :] -= factor * augmented_mat[row, :]

    x = retroactive_resolution(
        augmented_mat[:, 0:columns], augmented_mat[:, columns : columns + 1]
    )

    return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## In Static Equilibrium
"""
Checks if a system of forces is in static equilibrium.
"""
#from __future__ import annotations

from numpy import array, cos, cross, float64, radians, sin
from numpy.typing import NDArray


def polar_force(
    magnitude: float, angle: float, radian_mode: bool = False
) -> list[float]:
    """
    Resolves force along rectangular components.
    (force, angle) => (force_x, force_y)
    >>> import math
    >>> force = polar_force(10, 45)
    >>> math.isclose(force[0], 7.071067811865477)
    True
    >>> math.isclose(force[1], 7.0710678118654755)
    True
    >>> force = polar_force(10, 3.14, radian_mode=True)
    >>> math.isclose(force[0], -9.999987317275396)
    True
    >>> math.isclose(force[1], 0.01592652916486828)
    True
    """
    if radian_mode:
        return [magnitude * cos(angle), magnitude * sin(angle)]
    return [magnitude * cos(radians(angle)), magnitude * sin(radians(angle))]


def in_static_equilibrium(
    forces: NDArray[float64], location: NDArray[float64], eps: float = 10**-1
) -> bool:
    """
    Check if a system is in equilibrium.
    It takes two numpy.array objects.
    forces ==>  [
                        [force1_x, force1_y],
                        [force2_x, force2_y],
                        ....]
    location ==>  [
                        [x1, y1],
                        [x2, y2],
                        ....]
    >>> force = array([[1, 1], [-1, 2]])
    >>> location = array([[1, 0], [10, 0]])
    >>> in_static_equilibrium(force, location)
    False
    """
    # summation of moments is zero
    moments: NDArray[float64] = cross(location, forces)
    sum_moments: float = sum(moments)
    return abs(sum_moments) < eps


if __name__ == "__main__":
    # Test to check if it works
    forces = array(
        [
            polar_force(718.4, 180 - 30),
            polar_force(879.54, 45),
            polar_force(100, -90),
        ]
    )

    location: NDArray[float64] = array([[0, 0], [0, 0], [0, 0]])

    assert in_static_equilibrium(forces, location)

    # Problem 1 in image_data/2D_problems.jpg
    forces = array(
        [
            polar_force(30 * 9.81, 15),
            polar_force(215, 180 - 45),
            polar_force(264, 90 - 30),
        ]
    )

    location = array([[0, 0], [0, 0], [0, 0]])

    assert in_static_equilibrium(forces, location)

    # Problem in image_data/2D_problems_1.jpg
    forces = array([[0, -2000], [0, -1200], [0, 15600], [0, -12400]])

    location = array([[0, 0], [6, 0], [10, 0], [12, 0]])

    assert in_static_equilibrium(forces, location)

    import doctest

    doctest.testmod()

## Intersection
import math
from collections.abc import Callable


def intersection(function: Callable[[float], float], x0: float, x1: float) -> float:
    """
    function is the f we want to find its root
    x0 and x1 are two random starting points
    >>> intersection(lambda x: x ** 3 - 1, -5, 5)
    0.9999999999954654
    >>> intersection(lambda x: x ** 3 - 1, 5, 5)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: float division by zero, could not find root
    >>> intersection(lambda x: x ** 3 - 1, 100, 200)
    1.0000000000003888
    >>> intersection(lambda x: x ** 2 - 4 * x + 3, 0, 2)
    0.9999999998088019
    >>> intersection(lambda x: x ** 2 - 4 * x + 3, 2, 4)
    2.9999999998088023
    >>> intersection(lambda x: x ** 2 - 4 * x + 3, 4, 1000)
    3.0000000001786042
    >>> intersection(math.sin, -math.pi, math.pi)
    0.0
    >>> intersection(math.cos, -math.pi, math.pi)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: float division by zero, could not find root
    """
    x_n: float = x0
    x_n1: float = x1
    while True:
        if x_n == x_n1 or function(x_n1) == function(x_n):
            raise ZeroDivisionError("float division by zero, could not find root")
        x_n2: float = x_n1 - (
            function(x_n1) / ((function(x_n1) - function(x_n)) / (x_n1 - x_n))
        )
        if abs(x_n2 - x_n1) < 10**-5:
            return x_n2
        x_n = x_n1
        x_n1 = x_n2


def f(x: float) -> float:
    return math.pow(x, 3) - (2 * x) - 5


if __name__ == "__main__":
    print(intersection(f, 3, 3.5))

## Jacobi Iteration Method
"""
Jacobi Iteration Method - https://en.wikipedia.org/wiki/Jacobi_method
"""
#from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray


# Method to find solution of system of linear equations
def jacobi_iteration_method(
    coefficient_matrix: NDArray[float64],
    constant_matrix: NDArray[float64],
    init_val: list[int],
    iterations: int,
) -> list[float]:
    """
    Jacobi Iteration Method:
    An iterative algorithm to determine the solutions of strictly diagonally dominant
    system of linear equations

    4x1 +  x2 +  x3 =  2
     x1 + 5x2 + 2x3 = -6
     x1 + 2x2 + 4x3 = -4

    x_init = [0.5, -0.5 , -0.5]

    Examples:

    >>> coefficient = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 4]])
    >>> constant = np.array([[2], [-6], [-4]])
    >>> init_val = [0.5, -0.5, -0.5]
    >>> iterations = 3
    >>> jacobi_iteration_method(coefficient, constant, init_val, iterations)
    [0.909375, -1.14375, -0.7484375]


    >>> coefficient = np.array([[4, 1, 1], [1, 5, 2]])
    >>> constant = np.array([[2], [-6], [-4]])
    >>> init_val = [0.5, -0.5, -0.5]
    >>> iterations = 3
    >>> jacobi_iteration_method(coefficient, constant, init_val, iterations)
    Traceback (most recent call last):
        ...
    ValueError: Coefficient matrix dimensions must be nxn but received 2x3

    >>> coefficient = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 4]])
    >>> constant = np.array([[2], [-6]])
    >>> init_val = [0.5, -0.5, -0.5]
    >>> iterations = 3
    >>> jacobi_iteration_method(coefficient, constant, init_val, iterations)
    Traceback (most recent call last):
        ...
    ValueError: Coefficient and constant matrices dimensions must be nxn and nx1 but
                received 3x3 and 2x1

    >>> coefficient = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 4]])
    >>> constant = np.array([[2], [-6], [-4]])
    >>> init_val = [0.5, -0.5]
    >>> iterations = 3
    >>> jacobi_iteration_method(coefficient, constant, init_val, iterations)
    Traceback (most recent call last):
        ...
    ValueError: Number of initial values must be equal to number of rows in coefficient
                matrix but received 2 and 3

    >>> coefficient = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 4]])
    >>> constant = np.array([[2], [-6], [-4]])
    >>> init_val = [0.5, -0.5, -0.5]
    >>> iterations = 0
    >>> jacobi_iteration_method(coefficient, constant, init_val, iterations)
    Traceback (most recent call last):
        ...
    ValueError: Iterations must be at least 1
    """

    rows1, cols1 = coefficient_matrix.shape
    rows2, cols2 = constant_matrix.shape

    if rows1 != cols1:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received {rows1}x{cols1}"
        )

    if cols2 != 1:
        raise ValueError(f"Constant matrix must be nx1 but received {rows2}x{cols2}")

    if rows1 != rows2:
        raise ValueError(
            f"""Coefficient and constant matrices dimensions must be nxn and nx1 but
            received {rows1}x{cols1} and {rows2}x{cols2}"""
        )

    if len(init_val) != rows1:
        raise ValueError(
            f"""Number of initial values must be equal to number of rows in coefficient
            matrix but received {len(init_val)} and {rows1}"""
        )

    if iterations <= 0:
        raise ValueError("Iterations must be at least 1")

    table: NDArray[float64] = np.concatenate(
        (coefficient_matrix, constant_matrix), axis=1
    )

    rows, cols = table.shape

    strictly_diagonally_dominant(table)

    # Iterates the whole matrix for given number of times
    for _ in range(iterations):
        new_val = []
        for row in range(rows):
            temp = 0
            for col in range(cols):
                if col == row:
                    denom = table[row][col]
                elif col == cols - 1:
                    val = table[row][col]
                else:
                    temp += (-1) * table[row][col] * init_val[col]
            temp = (temp + val) / denom
            new_val.append(temp)
        init_val = new_val

    return [float(i) for i in new_val]


# Checks if the given matrix is strictly diagonally dominant
def strictly_diagonally_dominant(table: NDArray[float64]) -> bool:
    """
    >>> table = np.array([[4, 1, 1, 2], [1, 5, 2, -6], [1, 2, 4, -4]])
    >>> strictly_diagonally_dominant(table)
    True

    >>> table = np.array([[4, 1, 1, 2], [1, 5, 2, -6], [1, 2, 3, -4]])
    >>> strictly_diagonally_dominant(table)
    Traceback (most recent call last):
        ...
    ValueError: Coefficient matrix is not strictly diagonally dominant
    """

    rows, cols = table.shape

    is_diagonally_dominant = True

    for i in range(0, rows):
        total = 0
        for j in range(0, cols - 1):
            if i == j:
                continue
            else:
                total += table[i][j]

        if table[i][i] <= total:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return is_diagonally_dominant


# Test Cases
if __name__ == "__main__":
    import doctest

    doctest.testmod()

#####

### *** Backtracking
## All Combinations
"""
        In this problem, we want to determine all possible combinations of k
        numbers out of 1 ... n. We use backtracking to solve this problem.
        Time complexity: O(C(n,k)) which is O(n choose k) = O((n!/(k! * (n - k)!)))
"""
#from __future__ import annotations


def generate_all_combinations(n: int, k: int) -> list[list[int]]:
    """
    >>> generate_all_combinations(n=4, k=2)
    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    """

    result: list[list[int]] = []
    create_all_state(1, n, k, [], result)
    return result


def create_all_state(
    increment: int,
    total_number: int,
    level: int,
    current_list: list[int],
    total_list: list[list[int]],
) -> None:
    if level == 0:
        total_list.append(current_list[:])
        return

    for i in range(increment, total_number - level + 2):
        current_list.append(i)
        create_all_state(i + 1, total_number, level - 1, current_list, total_list)
        current_list.pop()


def print_all_state(total_list: list[list[int]]) -> None:
    for i in total_list:
        print(*i)


if __name__ == "__main__":
    n = 4
    k = 2
    total_list = generate_all_combinations(n, k)
    print_all_state(total_list)

## All Permutations
"""
        In this problem, we want to determine all possible permutations
        of the given sequence. We use backtracking to solve this problem.

        Time complexity: O(n! * n),
        where n denotes the length of the given sequence.
"""
#from __future__ import annotations


def generate_all_permutations(sequence: list[int | str]) -> None:
    create_state_space_tree(sequence, [], 0, [0 for i in range(len(sequence))])


def create_state_space_tree(
    sequence: list[int | str],
    current_sequence: list[int | str],
    index: int,
    index_used: list[int],
) -> None:
    """
    Creates a state space tree to iterate through each branch using DFS.
    We know that each state has exactly len(sequence) - index children.
    It terminates when it reaches the end of the given sequence.
    """

    if index == len(sequence):
        print(current_sequence)
        return

    for i in range(len(sequence)):
        if not index_used[i]:
            current_sequence.append(sequence[i])
            index_used[i] = True
            create_state_space_tree(sequence, current_sequence, index + 1, index_used)
            current_sequence.pop()
            index_used[i] = False


"""
remove the comment to take an input from the user

print("Enter the elements")
sequence = list(map(int, input().split()))
"""

sequence: list[int | str] = [3, 1, 2, 4]
generate_all_permutations(sequence)

sequence_2: list[int | str] = ["A", "B", "C"]
generate_all_permutations(sequence_2)

## All Subsequences
"""
In this problem, we want to determine all possible subsequences
of the given sequence. We use backtracking to solve this problem.

Time complexity: O(2^n),
where n denotes the length of the given sequence.
"""
#from __future__ import annotations

from typing import Any


def generate_all_subsequences(sequence: list[Any]) -> None:
    create_state_space_tree(sequence, [], 0)


def create_state_space_tree(
    sequence: list[Any], current_subsequence: list[Any], index: int
) -> None:
    """
    Creates a state space tree to iterate through each branch using DFS.
    We know that each state has exactly two children.
    It terminates when it reaches the end of the given sequence.
    """

    if index == len(sequence):
        print(current_subsequence)
        return

    create_state_space_tree(sequence, current_subsequence, index + 1)
    current_subsequence.append(sequence[index])
    create_state_space_tree(sequence, current_subsequence, index + 1)
    current_subsequence.pop()


if __name__ == "__main__":
    seq: list[Any] = [3, 1, 2, 4]
    generate_all_subsequences(seq)

    seq.clear()
    seq.extend(["A", "B", "C"])
    generate_all_subsequences(seq)

#####

### *** Conversions
## Binary To Decimal
def bin_to_decimal(bin_string: str) -> int:
    """
    Convert a binary value to its decimal equivalent

    >>> bin_to_decimal("101")
    5
    >>> bin_to_decimal(" 1010   ")
    10
    >>> bin_to_decimal("-11101")
    -29
    >>> bin_to_decimal("0")
    0
    >>> bin_to_decimal("a")
    Traceback (most recent call last):
        ...
    ValueError: Non-binary value was passed to the function
    >>> bin_to_decimal("")
    Traceback (most recent call last):
        ...
    ValueError: Empty string was passed to the function
    >>> bin_to_decimal("39")
    Traceback (most recent call last):
        ...
    ValueError: Non-binary value was passed to the function
    """
    bin_string = str(bin_string).strip()
    if not bin_string:
        raise ValueError("Empty string was passed to the function")
    is_negative = bin_string[0] == "-"
    if is_negative:
        bin_string = bin_string[1:]
    if not all(char in "01" for char in bin_string):
        raise ValueError("Non-binary value was passed to the function")
    decimal_number = 0
    for char in bin_string:
        decimal_number = 2 * decimal_number + int(char)
    return -decimal_number if is_negative else decimal_number


if __name__ == "__main__":
    from doctest import testmod

    testmod()

## Binary To Hexadecimal
BITS_TO_HEX = {
    "0000": "0",
    "0001": "1",
    "0010": "2",
    "0011": "3",
    "0100": "4",
    "0101": "5",
    "0110": "6",
    "0111": "7",
    "1000": "8",
    "1001": "9",
    "1010": "a",
    "1011": "b",
    "1100": "c",
    "1101": "d",
    "1110": "e",
    "1111": "f",
}


def bin_to_hexadecimal(binary_str: str) -> str:
    """
    Converting a binary string into hexadecimal using Grouping Method

    >>> bin_to_hexadecimal('101011111')
    '0x15f'
    >>> bin_to_hexadecimal(' 1010   ')
    '0x0a'
    >>> bin_to_hexadecimal('-11101')
    '-0x1d'
    >>> bin_to_hexadecimal('a')
    Traceback (most recent call last):
        ...
    ValueError: Non-binary value was passed to the function
    >>> bin_to_hexadecimal('')
    Traceback (most recent call last):
        ...
    ValueError: Empty string was passed to the function
    """
    # Sanitising parameter
    binary_str = str(binary_str).strip()

    # Exceptions
    if not binary_str:
        raise ValueError("Empty string was passed to the function")
    is_negative = binary_str[0] == "-"
    binary_str = binary_str[1:] if is_negative else binary_str
    if not all(char in "01" for char in binary_str):
        raise ValueError("Non-binary value was passed to the function")

    binary_str = (
        "0" * (4 * (divmod(len(binary_str), 4)[0] + 1) - len(binary_str)) + binary_str
    )

    hexadecimal = []
    for x in range(0, len(binary_str), 4):
        hexadecimal.append(BITS_TO_HEX[binary_str[x : x + 4]])
    hexadecimal_str = "0x" + "".join(hexadecimal)

    return "-" + hexadecimal_str if is_negative else hexadecimal_str


if __name__ == "__main__":
    from doctest import testmod

    testmod()

## Binary To Octal
"""
The function below will convert any binary string to the octal equivalent.

>>> bin_to_octal("1111")
'17'

>>> bin_to_octal("101010101010011")
'52523'

>>> bin_to_octal("")
Traceback (most recent call last):
    ...
ValueError: Empty string was passed to the function
>>> bin_to_octal("a-1")
Traceback (most recent call last):
    ...
ValueError: Non-binary value was passed to the function
"""


def bin_to_octal(bin_string: str) -> str:
    if not all(char in "01" for char in bin_string):
        raise ValueError("Non-binary value was passed to the function")
    if not bin_string:
        raise ValueError("Empty string was passed to the function")
    oct_string = ""
    while len(bin_string) % 3 != 0:
        bin_string = "0" + bin_string
    bin_string_in_3_list = [
        bin_string[index : index + 3]
        for index in range(len(bin_string))
        if index % 3 == 0
    ]
    for bin_group in bin_string_in_3_list:
        oct_val = 0
        for index, val in enumerate(bin_group):
            oct_val += int(2 ** (2 - index) * int(val))
        oct_string += str(oct_val)
    return oct_string


if __name__ == "__main__":
    from doctest import testmod

    testmod()

## Decimal To Any
"""Convert a positive Decimal Number to Any Other Representation"""

from string import ascii_uppercase

ALPHABET_VALUES = {str(ord(c) - 55): c for c in ascii_uppercase}


def decimal_to_any(num: int, base: int) -> str:
    """
    Convert a positive integer to another base as str.
    >>> decimal_to_any(0, 2)
    '0'
    >>> decimal_to_any(5, 4)
    '11'
    >>> decimal_to_any(20, 3)
    '202'
    >>> decimal_to_any(58, 16)
    '3A'
    >>> decimal_to_any(243, 17)
    'E5'
    >>> decimal_to_any(34923, 36)
    'QY3'
    >>> decimal_to_any(10, 11)
    'A'
    >>> decimal_to_any(16, 16)
    '10'
    >>> decimal_to_any(36, 36)
    '10'
    >>> # negatives will error
    >>> decimal_to_any(-45, 8)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: parameter must be positive int
    >>> # floats will error
    >>> decimal_to_any(34.4, 6) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: int() can't convert non-string with explicit base
    >>> # a float base will error
    >>> decimal_to_any(5, 2.5) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: 'float' object cannot be interpreted as an integer
    >>> # a str base will error
    >>> decimal_to_any(10, '16') # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: 'str' object cannot be interpreted as an integer
    >>> # a base less than 2 will error
    >>> decimal_to_any(7, 0) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: base must be >= 2
    >>> # a base greater than 36 will error
    >>> decimal_to_any(34, 37) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: base must be <= 36
    """
    if isinstance(num, float):
        raise TypeError("int() can't convert non-string with explicit base")
    if num < 0:
        raise ValueError("parameter must be positive int")
    if isinstance(base, str):
        raise TypeError("'str' object cannot be interpreted as an integer")
    if isinstance(base, float):
        raise TypeError("'float' object cannot be interpreted as an integer")
    if base in (0, 1):
        raise ValueError("base must be >= 2")
    if base > 36:
        raise ValueError("base must be <= 36")
    new_value = ""
    mod = 0
    div = 0
    while div != 1:
        div, mod = divmod(num, base)
        if base >= 11 and 9 < mod < 36:
            actual_value = ALPHABET_VALUES[str(mod)]
        else:
            actual_value = str(mod)
        new_value += actual_value
        div = num // base
        num = div
        if div == 0:
            return str(new_value[::-1])
        elif div == 1:
            new_value += str(div)
            return str(new_value[::-1])

    return new_value[::-1]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    for base in range(2, 37):
        for num in range(1000):
            assert int(decimal_to_any(num, base), base) == num, (
                num,
                base,
                decimal_to_any(num, base),
                int(decimal_to_any(num, base), base),
            )

## Decimal To Binary
"""Convert a Decimal Number to a Binary Number."""


def decimal_to_binary(num: int) -> str:
    """
    Convert an Integer Decimal Number to a Binary Number as str.
    >>> decimal_to_binary(0)
    '0b0'
    >>> decimal_to_binary(2)
    '0b10'
    >>> decimal_to_binary(7)
    '0b111'
    >>> decimal_to_binary(35)
    '0b100011'
    >>> # negatives work too
    >>> decimal_to_binary(-2)
    '-0b10'
    >>> # other floats will error
    >>> decimal_to_binary(16.16) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: 'float' object cannot be interpreted as an integer
    >>> # strings will error as well
    >>> decimal_to_binary('0xfffff') # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: 'str' object cannot be interpreted as an integer
    """

    if isinstance(num, float):
        raise TypeError("'float' object cannot be interpreted as an integer")
    if isinstance(num, str):
        raise TypeError("'str' object cannot be interpreted as an integer")

    if num == 0:
        return "0b0"

    negative = False

    if num < 0:
        negative = True
        num = -num

    binary: list[int] = []
    while num > 0:
        binary.insert(0, num % 2)
        num >>= 1

    if negative:
        return "-0b" + "".join(str(e) for e in binary)

    return "0b" + "".join(str(e) for e in binary)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Decimal To Binary Recursion
def binary_recursive(decimal: int) -> str:
    """
    Take a positive integer value and return its binary equivalent.
    >>> binary_recursive(1000)
    '1111101000'
    >>> binary_recursive("72")
    '1001000'
    >>> binary_recursive("number")
    Traceback (most recent call last):
        ...
    ValueError: invalid literal for int() with base 10: 'number'
    """
    decimal = int(decimal)
    if decimal in (0, 1):  # Exit cases for the recursion
        return str(decimal)
    div, mod = divmod(decimal, 2)
    return binary_recursive(div) + str(mod)


def main(number: str) -> str:
    """
    Take an integer value and raise ValueError for wrong inputs,
    call the function above and return the output with prefix "0b" & "-0b"
    for positive and negative integers respectively.
    >>> main(0)
    '0b0'
    >>> main(40)
    '0b101000'
    >>> main(-40)
    '-0b101000'
    >>> main(40.8)
    Traceback (most recent call last):
        ...
    ValueError: Input value is not an integer
    >>> main("forty")
    Traceback (most recent call last):
        ...
    ValueError: Input value is not an integer
    """
    number = str(number).strip()
    if not number:
        raise ValueError("No input value was provided")
    negative = "-" if number.startswith("-") else ""
    number = number.lstrip("-")
    if not number.isnumeric():
        raise ValueError("Input value is not an integer")
    return f"{negative}0b{binary_recursive(int(number))}"


if __name__ == "__main__":
    from doctest import testmod

    testmod()

## Decimal To Hexadecimal
""" Convert Base 10 (Decimal) Values to Hexadecimal Representations """

# set decimal value for each hexadecimal digit
values = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "a",
    11: "b",
    12: "c",
    13: "d",
    14: "e",
    15: "f",
}


def decimal_to_hexadecimal(decimal: float) -> str:
    """
    take integer decimal value, return hexadecimal representation as str beginning
    with 0x
    >>> decimal_to_hexadecimal(5)
    '0x5'
    >>> decimal_to_hexadecimal(15)
    '0xf'
    >>> decimal_to_hexadecimal(37)
    '0x25'
    >>> decimal_to_hexadecimal(255)
    '0xff'
    >>> decimal_to_hexadecimal(4096)
    '0x1000'
    >>> decimal_to_hexadecimal(999098)
    '0xf3eba'
    >>> # negatives work too
    >>> decimal_to_hexadecimal(-256)
    '-0x100'
    >>> # floats are acceptable if equivalent to an int
    >>> decimal_to_hexadecimal(17.0)
    '0x11'
    >>> # other floats will error
    >>> decimal_to_hexadecimal(16.16) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    AssertionError
    >>> # strings will error as well
    >>> decimal_to_hexadecimal('0xfffff') # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    AssertionError
    >>> # results are the same when compared to Python's default hex function
    >>> decimal_to_hexadecimal(-256) == hex(-256)
    True
    """
    assert type(decimal) in (int, float) and decimal == int(decimal)
    decimal = int(decimal)
    hexadecimal = ""
    negative = False
    if decimal < 0:
        negative = True
        decimal *= -1
    while decimal > 0:
        decimal, remainder = divmod(decimal, 16)
        hexadecimal = values[remainder] + hexadecimal
    hexadecimal = "0x" + hexadecimal
    if negative:
        hexadecimal = "-" + hexadecimal
    return hexadecimal


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Decimal To Octal
"""Convert a Decimal Number to an Octal Number."""

import math

# Modified from:
# https://github.com/TheAlgorithms/Javascript/blob/master/Conversions/DecimalToOctal.js


def decimal_to_octal(num: int) -> str:
    """Convert a Decimal Number to an Octal Number.

    >>> all(decimal_to_octal(i) == oct(i) for i
    ...     in (0, 2, 8, 64, 65, 216, 255, 256, 512))
    True
    """
    octal = 0
    counter = 0
    while num > 0:
        remainder = num % 8
        octal = octal + (remainder * math.floor(math.pow(10, counter)))
        counter += 1
        num = math.floor(num / 8)  # basically /= 8 without remainder if any
        # This formatting removes trailing '.0' from `octal`.
    return f"0o{int(octal)}"


def main() -> None:
    """Print octal equivalents of decimal numbers."""
    print("\n2 in octal is:")
    print(decimal_to_octal(2))  # = 2
    print("\n8 in octal is:")
    print(decimal_to_octal(8))  # = 10
    print("\n65 in octal is:")
    print(decimal_to_octal(65))  # = 101
    print("\n216 in octal is:")
    print(decimal_to_octal(216))  # = 330
    print("\n512 in octal is:")
    print(decimal_to_octal(512))  # = 1000
    print("\n")


if __name__ == "__main__":
    main()

### *** Searches
## Binary Search
#!/usr/bin/env python3

"""
This is pure Python implementation of binary search algorithms

For doctests run following command:
python3 -m doctest -v binary_search.py

For manual testing run:
python3 binary_search.py
"""
#from __future__ import annotations

import bisect


def bisect_left(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> int:
    """
    Locates the first element in a sorted array that is larger or equal to a given
    value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.bisect_left .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to bisect
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])
    :return: index i such that all values in sorted_collection[lo:i] are < item and all
        values in sorted_collection[i:hi] are >= item.

    Examples:
    >>> bisect_left([0, 5, 7, 10, 15], 0)
    0

    >>> bisect_left([0, 5, 7, 10, 15], 6)
    2

    >>> bisect_left([0, 5, 7, 10, 15], 20)
    5

    >>> bisect_left([0, 5, 7, 10, 15], 15, 1, 3)
    3

    >>> bisect_left([0, 5, 7, 10, 15], 6, 2)
    2
    """
    if hi < 0:
        hi = len(sorted_collection)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sorted_collection[mid] < item:
            lo = mid + 1
        else:
            hi = mid

    return lo


def bisect_right(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> int:
    """
    Locates the first element in a sorted array that is larger than a given value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.bisect_right .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to bisect
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])
    :return: index i such that all values in sorted_collection[lo:i] are <= item and
        all values in sorted_collection[i:hi] are > item.

    Examples:
    >>> bisect_right([0, 5, 7, 10, 15], 0)
    1

    >>> bisect_right([0, 5, 7, 10, 15], 15)
    5

    >>> bisect_right([0, 5, 7, 10, 15], 6)
    2

    >>> bisect_right([0, 5, 7, 10, 15], 15, 1, 3)
    3

    >>> bisect_right([0, 5, 7, 10, 15], 6, 2)
    2
    """
    if hi < 0:
        hi = len(sorted_collection)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if sorted_collection[mid] <= item:
            lo = mid + 1
        else:
            hi = mid

    return lo


def insort_left(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> None:
    """
    Inserts a given value into a sorted array before other values with the same value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.insort_left .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to insert
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])

    Examples:
    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_left(sorted_collection, 6)
    >>> sorted_collection
    [0, 5, 6, 7, 10, 15]

    >>> sorted_collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item = (5, 5)
    >>> insort_left(sorted_collection, item)
    >>> sorted_collection
    [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item is sorted_collection[1]
    True
    >>> item is sorted_collection[2]
    False

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_left(sorted_collection, 20)
    >>> sorted_collection
    [0, 5, 7, 10, 15, 20]

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_left(sorted_collection, 15, 1, 3)
    >>> sorted_collection
    [0, 5, 7, 15, 10, 15]
    """
    sorted_collection.insert(bisect_left(sorted_collection, item, lo, hi), item)


def insort_right(
    sorted_collection: list[int], item: int, lo: int = 0, hi: int = -1
) -> None:
    """
    Inserts a given value into a sorted array after other values with the same value.

    It has the same interface as
    https://docs.python.org/3/library/bisect.html#bisect.insort_right .

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item to insert
    :param lo: lowest index to consider (as in sorted_collection[lo:hi])
    :param hi: past the highest index to consider (as in sorted_collection[lo:hi])

    Examples:
    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_right(sorted_collection, 6)
    >>> sorted_collection
    [0, 5, 6, 7, 10, 15]

    >>> sorted_collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item = (5, 5)
    >>> insort_right(sorted_collection, item)
    >>> sorted_collection
    [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    >>> item is sorted_collection[1]
    False
    >>> item is sorted_collection[2]
    True

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_right(sorted_collection, 20)
    >>> sorted_collection
    [0, 5, 7, 10, 15, 20]

    >>> sorted_collection = [0, 5, 7, 10, 15]
    >>> insort_right(sorted_collection, 15, 1, 3)
    >>> sorted_collection
    [0, 5, 7, 15, 10, 15]
    """
    sorted_collection.insert(bisect_right(sorted_collection, item, lo, hi), item)


def binary_search(sorted_collection: list[int], item: int) -> int | None:
    """Pure implementation of binary search algorithm in Python

    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> binary_search([0, 5, 7, 10, 15], 0)
    0

    >>> binary_search([0, 5, 7, 10, 15], 15)
    4

    >>> binary_search([0, 5, 7, 10, 15], 5)
    1

    >>> binary_search([0, 5, 7, 10, 15], 6)

    """
    left = 0
    right = len(sorted_collection) - 1

    while left <= right:
        midpoint = left + (right - left) // 2
        current_item = sorted_collection[midpoint]
        if current_item == item:
            return midpoint
        elif item < current_item:
            right = midpoint - 1
        else:
            left = midpoint + 1
    return None


def binary_search_std_lib(sorted_collection: list[int], item: int) -> int | None:
    """Pure implementation of binary search algorithm in Python using stdlib

    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> binary_search_std_lib([0, 5, 7, 10, 15], 0)
    0

    >>> binary_search_std_lib([0, 5, 7, 10, 15], 15)
    4

    >>> binary_search_std_lib([0, 5, 7, 10, 15], 5)
    1

    >>> binary_search_std_lib([0, 5, 7, 10, 15], 6)

    """
    index = bisect.bisect_left(sorted_collection, item)
    if index != len(sorted_collection) and sorted_collection[index] == item:
        return index
    return None


def binary_search_by_recursion(
    sorted_collection: list[int], item: int, left: int, right: int
) -> int | None:
    """Pure implementation of binary search algorithm in Python by recursion

    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable
    First recursion should be started with left=0 and right=(len(sorted_collection)-1)

    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 0, 0, 4)
    0

    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 15, 0, 4)
    4

    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 5, 0, 4)
    1

    >>> binary_search_by_recursion([0, 5, 7, 10, 15], 6, 0, 4)

    """
    if right < left:
        return None

    midpoint = left + (right - left) // 2

    if sorted_collection[midpoint] == item:
        return midpoint
    elif sorted_collection[midpoint] > item:
        return binary_search_by_recursion(sorted_collection, item, left, midpoint - 1)
    else:
        return binary_search_by_recursion(sorted_collection, item, midpoint + 1, right)


if __name__ == "__main__":
    user_input = input("Enter numbers separated by comma:\n").strip()
    collection = sorted(int(item) for item in user_input.split(","))
    target = int(input("Enter a single number to be found in the list:\n"))
    result = binary_search(collection, target)
    if result is None:
        print(f"{target} was not found in {collection}.")
    else:
        print(f"{target} was found at position {result} in {collection}.")

## Binary Tree Traversal
"""
This is pure Python implementation of tree traversal algorithms
"""
#from __future__ import annotations

import queue


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None


def build_tree():
    print("\n********Press N to stop entering at any point of time********\n")
    check = input("Enter the value of the root node: ").strip().lower() or "n"
    if check == "n":
        return None
    q: queue.Queue = queue.Queue()
    tree_node = TreeNode(int(check))
    q.put(tree_node)
    while not q.empty():
        node_found = q.get()
        msg = f"Enter the left node of {node_found.data}: "
        check = input(msg).strip().lower() or "n"
        if check == "n":
            return tree_node
        left_node = TreeNode(int(check))
        node_found.left = left_node
        q.put(left_node)
        msg = f"Enter the right node of {node_found.data}: "
        check = input(msg).strip().lower() or "n"
        if check == "n":
            return tree_node
        right_node = TreeNode(int(check))
        node_found.right = right_node
        q.put(right_node)
    return None


def pre_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> pre_order(root)
    1,2,4,5,3,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    print(node.data, end=",")
    pre_order(node.left)
    pre_order(node.right)


def in_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> in_order(root)
    4,2,5,1,6,3,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    in_order(node.left)
    print(node.data, end=",")
    in_order(node.right)


def post_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> post_order(root)
    4,5,2,6,7,3,1,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    post_order(node.left)
    post_order(node.right)
    print(node.data, end=",")


def level_order(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> level_order(root)
    1,2,3,4,5,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    q: queue.Queue = queue.Queue()
    q.put(node)
    while not q.empty():
        node_dequeued = q.get()
        print(node_dequeued.data, end=",")
        if node_dequeued.left:
            q.put(node_dequeued.left)
        if node_dequeued.right:
            q.put(node_dequeued.right)


def level_order_actual(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> level_order_actual(root)
    1,
    2,3,
    4,5,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    q: queue.Queue = queue.Queue()
    q.put(node)
    while not q.empty():
        list_ = []
        while not q.empty():
            node_dequeued = q.get()
            print(node_dequeued.data, end=",")
            if node_dequeued.left:
                list_.append(node_dequeued.left)
            if node_dequeued.right:
                list_.append(node_dequeued.right)
        print()
        for node in list_:
            q.put(node)


# iteration version
def pre_order_iter(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> pre_order_iter(root)
    1,2,4,5,3,6,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    stack: list[TreeNode] = []
    n = node
    while n or stack:
        while n:  # start from root node, find its left child
            print(n.data, end=",")
            stack.append(n)
            n = n.left
        # end of while means current node doesn't have left child
        n = stack.pop()
        # start to traverse its right child
        n = n.right


def in_order_iter(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> in_order_iter(root)
    4,2,5,1,6,3,7,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    stack: list[TreeNode] = []
    n = node
    while n or stack:
        while n:
            stack.append(n)
            n = n.left
        n = stack.pop()
        print(n.data, end=",")
        n = n.right


def post_order_iter(node: TreeNode) -> None:
    """
    >>> root = TreeNode(1)
    >>> tree_node2 = TreeNode(2)
    >>> tree_node3 = TreeNode(3)
    >>> tree_node4 = TreeNode(4)
    >>> tree_node5 = TreeNode(5)
    >>> tree_node6 = TreeNode(6)
    >>> tree_node7 = TreeNode(7)
    >>> root.left, root.right = tree_node2, tree_node3
    >>> tree_node2.left, tree_node2.right = tree_node4 , tree_node5
    >>> tree_node3.left, tree_node3.right = tree_node6 , tree_node7
    >>> post_order_iter(root)
    4,5,2,6,7,3,1,
    """
    if not isinstance(node, TreeNode) or not node:
        return
    stack1, stack2 = [], []
    n = node
    stack1.append(n)
    while stack1:  # to find the reversed order of post order, store it in stack2
        n = stack1.pop()
        if n.left:
            stack1.append(n.left)
        if n.right:
            stack1.append(n.right)
        stack2.append(n)
    while stack2:  # pop up from stack2 will be the post order
        print(stack2.pop().data, end=",")


def prompt(s: str = "", width=50, char="*") -> str:
    if not s:
        return "\n" + width * char
    left, extra = divmod(width - len(s) - 2, 2)
    return f"{left * char} {s} {(left + extra) * char}"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(prompt("Binary Tree Traversals"))

    node = build_tree()
    print(prompt("Pre Order Traversal"))
    pre_order(node)
    print(prompt() + "\n")

    print(prompt("In Order Traversal"))
    in_order(node)
    print(prompt() + "\n")

    print(prompt("Post Order Traversal"))
    post_order(node)
    print(prompt() + "\n")

    print(prompt("Level Order Traversal"))
    level_order(node)
    print(prompt() + "\n")

    print(prompt("Actual Level Order Traversal"))
    level_order_actual(node)
    print("*" * 50 + "\n")

    print(prompt("Pre Order Traversal - Iteration Version"))
    pre_order_iter(node)
    print(prompt() + "\n")

    print(prompt("In Order Traversal - Iteration Version"))
    in_order_iter(node)
    print(prompt() + "\n")

    print(prompt("Post Order Traversal - Iteration Version"))
    post_order_iter(node)
    print(prompt())

## Double Linear Search
#from __future__ import annotations


def double_linear_search(array: list[int], search_item: int) -> int:
    """
    Iterate through the array from both sides to find the index of search_item.

    :param array: the array to be searched
    :param search_item: the item to be searched
    :return the index of search_item, if search_item is in array, else -1

    Examples:
    >>> double_linear_search([1, 5, 5, 10], 1)
    0
    >>> double_linear_search([1, 5, 5, 10], 5)
    1
    >>> double_linear_search([1, 5, 5, 10], 100)
    -1
    >>> double_linear_search([1, 5, 5, 10], 10)
    3
    """
    # define the start and end index of the given array
    start_ind, end_ind = 0, len(array) - 1
    while start_ind <= end_ind:
        if array[start_ind] == search_item:
            return start_ind
        elif array[end_ind] == search_item:
            return end_ind
        else:
            start_ind += 1
            end_ind -= 1
    # returns -1 if search_item is not found in array
    return -1


if __name__ == "__main__":
    print(double_linear_search(list(range(100)), 40))

## Double Linear Search Recursion
def search(list_data: list, key: int, left: int = 0, right: int = 0) -> int:
    """
    Iterate through the array to find the index of key using recursion.
    :param list_data: the list to be searched
    :param key: the key to be searched
    :param left: the index of first element
    :param right: the index of last element
    :return: the index of key value if found, -1 otherwise.

    >>> search(list(range(0, 11)), 5)
    5
    >>> search([1, 2, 4, 5, 3], 4)
    2
    >>> search([1, 2, 4, 5, 3], 6)
    -1
    >>> search([5], 5)
    0
    >>> search([], 1)
    -1
    """
    right = right or len(list_data) - 1
    if left > right:
        return -1
    elif list_data[left] == key:
        return left
    elif list_data[right] == key:
        return right
    else:
        return search(list_data, key, left + 1, right - 1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Fibonacci Search
"""
This is pure Python implementation of fibonacci search.

Resources used:
https://en.wikipedia.org/wiki/Fibonacci_search_technique

For doctests run following command:
python3 -m doctest -v fibonacci_search.py

For manual testing run:
python3 fibonacci_search.py
"""
from functools import lru_cache


@lru_cache
def fibonacci(k: int) -> int:
    """Finds fibonacci number in index k.

    Parameters
    ----------
    k :
        Index of fibonacci.

    Returns
    -------
    int
        Fibonacci number in position k.

    >>> fibonacci(0)
    0
    >>> fibonacci(2)
    1
    >>> fibonacci(5)
    5
    >>> fibonacci(15)
    610
    >>> fibonacci('a')
    Traceback (most recent call last):
    TypeError: k must be an integer.
    >>> fibonacci(-5)
    Traceback (most recent call last):
    ValueError: k integer must be greater or equal to zero.
    """
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 0:
        raise ValueError("k integer must be greater or equal to zero.")
    if k == 0:
        return 0
    elif k == 1:
        return 1
    else:
        return fibonacci(k - 1) + fibonacci(k - 2)


def fibonacci_search(arr: list, val: int) -> int:
    """A pure Python implementation of a fibonacci search algorithm.

    Parameters
    ----------
    arr
        List of sorted elements.
    val
        Element to search in list.

    Returns
    -------
    int
        The index of the element in the array.
        -1 if the element is not found.

    >>> fibonacci_search([4, 5, 6, 7], 4)
    0
    >>> fibonacci_search([4, 5, 6, 7], -10)
    -1
    >>> fibonacci_search([-18, 2], -18)
    0
    >>> fibonacci_search([5], 5)
    0
    >>> fibonacci_search(['a', 'c', 'd'], 'c')
    1
    >>> fibonacci_search(['a', 'c', 'd'], 'f')
    -1
    >>> fibonacci_search([], 1)
    -1
    >>> fibonacci_search([.1, .4 , 7], .4)
    1
    >>> fibonacci_search([], 9)
    -1
    >>> fibonacci_search(list(range(100)), 63)
    63
    >>> fibonacci_search(list(range(100)), 99)
    99
    >>> fibonacci_search(list(range(-100, 100, 3)), -97)
    1
    >>> fibonacci_search(list(range(-100, 100, 3)), 0)
    -1
    >>> fibonacci_search(list(range(-100, 100, 5)), 0)
    20
    >>> fibonacci_search(list(range(-100, 100, 5)), 95)
    39
    """
    len_list = len(arr)
    # Find m such that F_m >= n where F_i is the i_th fibonacci number.
    i = 0
    while True:
        if fibonacci(i) >= len_list:
            fibb_k = i
            break
        i += 1
    offset = 0
    while fibb_k > 0:
        index_k = min(
            offset + fibonacci(fibb_k - 1), len_list - 1
        )  # Prevent out of range
        item_k_1 = arr[index_k]
        if item_k_1 == val:
            return index_k
        elif val < item_k_1:
            fibb_k -= 1
        elif val > item_k_1:
            offset += fibonacci(fibb_k - 1)
            fibb_k -= 2
    else:
        return -1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Hill Climbing
# https://en.wikipedia.org/wiki/Hill_climbing
import math


class SearchProblem:
    """
    An interface to define search problems.
    The interface will be illustrated using the example of mathematical function.
    """

    def __init__(self, x: int, y: int, step_size: int, function_to_optimize):
        """
        The constructor of the search problem.

        x: the x coordinate of the current search state.
        y: the y coordinate of the current search state.
        step_size: size of the step to take when looking for neighbors.
        function_to_optimize: a function to optimize having the signature f(x, y).
        """
        self.x = x
        self.y = y
        self.step_size = step_size
        self.function = function_to_optimize

    def score(self) -> int:
        """
        Returns the output of the function called with current x and y coordinates.
        >>> def test_function(x, y):
        ...     return x + y
        >>> SearchProblem(0, 0, 1, test_function).score()  # 0 + 0 = 0
        0
        >>> SearchProblem(5, 7, 1, test_function).score()  # 5 + 7 = 12
        12
        """
        return self.function(self.x, self.y)

    def get_neighbors(self):
        """
        Returns a list of coordinates of neighbors adjacent to the current coordinates.

        Neighbors:
        | 0 | 1 | 2 |
        | 3 | _ | 4 |
        | 5 | 6 | 7 |
        """
        step_size = self.step_size
        return [
            SearchProblem(x, y, step_size, self.function)
            for x, y in (
                (self.x - step_size, self.y - step_size),
                (self.x - step_size, self.y),
                (self.x - step_size, self.y + step_size),
                (self.x, self.y - step_size),
                (self.x, self.y + step_size),
                (self.x + step_size, self.y - step_size),
                (self.x + step_size, self.y),
                (self.x + step_size, self.y + step_size),
            )
        ]

    def __hash__(self):
        """
        hash the string representation of the current search state.
        """
        return hash(str(self))

    def __eq__(self, obj):
        """
        Check if the 2 objects are equal.
        """
        if isinstance(obj, SearchProblem):
            return hash(str(self)) == hash(str(obj))
        return False

    def __str__(self):
        """
        string representation of the current search state.
        >>> str(SearchProblem(0, 0, 1, None))
        'x: 0 y: 0'
        >>> str(SearchProblem(2, 5, 1, None))
        'x: 2 y: 5'
        """
        return f"x: {self.x} y: {self.y}"


def hill_climbing(
    search_prob,
    find_max: bool = True,
    max_x: float = math.inf,
    min_x: float = -math.inf,
    max_y: float = math.inf,
    min_y: float = -math.inf,
    visualization: bool = False,
    max_iter: int = 10000,
) -> SearchProblem:
    """
    Implementation of the hill climbling algorithm.
    We start with a given state, find all its neighbors,
    move towards the neighbor which provides the maximum (or minimum) change.
    We keep doing this until we are at a state where we do not have any
    neighbors which can improve the solution.
        Args:
            search_prob: The search state at the start.
            find_max: If True, the algorithm should find the maximum else the minimum.
            max_x, min_x, max_y, min_y: the maximum and minimum bounds of x and y.
            visualization: If True, a matplotlib graph is displayed.
            max_iter: number of times to run the iteration.
        Returns a search state having the maximum (or minimum) score.
    """
    current_state = search_prob
    scores = []  # list to store the current score at each iteration
    iterations = 0
    solution_found = False
    visited = set()
    while not solution_found and iterations < max_iter:
        visited.add(current_state)
        iterations += 1
        current_score = current_state.score()
        scores.append(current_score)
        neighbors = current_state.get_neighbors()
        max_change = -math.inf
        min_change = math.inf
        next_state = None  # to hold the next best neighbor
        for neighbor in neighbors:
            if neighbor in visited:
                continue  # do not want to visit the same state again
            if (
                neighbor.x > max_x
                or neighbor.x < min_x
                or neighbor.y > max_y
                or neighbor.y < min_y
            ):
                continue  # neighbor outside our bounds
            change = neighbor.score() - current_score
            if find_max:  # finding max
                # going to direction with greatest ascent
                if change > max_change and change > 0:
                    max_change = change
                    next_state = neighbor
            else:  # finding min
                # to direction with greatest descent
                if change < min_change and change < 0:
                    min_change = change
                    next_state = neighbor
        if next_state is not None:
            # we found at least one neighbor which improved the current state
            current_state = next_state
        else:
            # since we have no neighbor that improves the solution we stop the search
            solution_found = True

    if visualization:
        from matplotlib import pyplot as plt

        plt.plot(range(iterations), scores)
        plt.xlabel("Iterations")
        plt.ylabel("Function values")
        plt.show()

    return current_state


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    def test_f1(x, y):
        return (x**2) + (y**2)

    # starting the problem with initial coordinates (3, 4)
    prob = SearchProblem(x=3, y=4, step_size=1, function_to_optimize=test_f1)
    local_min = hill_climbing(prob, find_max=False)
    print(
        "The minimum score for f(x, y) = x^2 + y^2 found via hill climbing: "
        f"{local_min.score()}"
    )

    # starting the problem with initial coordinates (12, 47)
    prob = SearchProblem(x=12, y=47, step_size=1, function_to_optimize=test_f1)
    local_min = hill_climbing(
        prob, find_max=False, max_x=100, min_x=5, max_y=50, min_y=-5, visualization=True
    )
    print(
        "The minimum score for f(x, y) = x^2 + y^2 with the domain 100 > x > 5 "
        f"and 50 > y > - 5 found via hill climbing: {local_min.score()}"
    )

    def test_f2(x, y):
        return (3 * x**2) - (6 * y)

    prob = SearchProblem(x=3, y=4, step_size=1, function_to_optimize=test_f1)
    local_min = hill_climbing(prob, find_max=True)
    print(
        "The maximum score for f(x, y) = x^2 + y^2 found via hill climbing: "
        f"{local_min.score()}"
    )

## Interpolation Search
"""
This is pure Python implementation of interpolation search algorithm
"""


def interpolation_search(sorted_collection, item):
    """Pure implementation of interpolation search algorithm in Python
    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable
    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found
    """
    left = 0
    right = len(sorted_collection) - 1

    while left <= right:
        # avoid divided by 0 during interpolation
        if sorted_collection[left] == sorted_collection[right]:
            if sorted_collection[left] == item:
                return left
            else:
                return None

        point = left + ((item - sorted_collection[left]) * (right - left)) // (
            sorted_collection[right] - sorted_collection[left]
        )

        # out of range check
        if point < 0 or point >= len(sorted_collection):
            return None

        current_item = sorted_collection[point]
        if current_item == item:
            return point
        else:
            if point < left:
                right = left
                left = point
            elif point > right:
                left = right
                right = point
            else:
                if item < current_item:
                    right = point - 1
                else:
                    left = point + 1
    return None


def interpolation_search_by_recursion(sorted_collection, item, left, right):
    """Pure implementation of interpolation search algorithm in Python by recursion
    Be careful collection must be ascending sorted, otherwise result will be
    unpredictable
    First recursion should be started with left=0 and right=(len(sorted_collection)-1)
    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found
    """

    # avoid divided by 0 during interpolation
    if sorted_collection[left] == sorted_collection[right]:
        if sorted_collection[left] == item:
            return left
        else:
            return None

    point = left + ((item - sorted_collection[left]) * (right - left)) // (
        sorted_collection[right] - sorted_collection[left]
    )

    # out of range check
    if point < 0 or point >= len(sorted_collection):
        return None

    if sorted_collection[point] == item:
        return point
    elif point < left:
        return interpolation_search_by_recursion(sorted_collection, item, point, left)
    elif point > right:
        return interpolation_search_by_recursion(sorted_collection, item, right, left)
    else:
        if sorted_collection[point] > item:
            return interpolation_search_by_recursion(
                sorted_collection, item, left, point - 1
            )
        else:
            return interpolation_search_by_recursion(
                sorted_collection, item, point + 1, right
            )


def __assert_sorted(collection):
    """Check if collection is ascending sorted, if not - raises :py:class:`ValueError`
    :param collection: collection
    :return: True if collection is ascending sorted
    :raise: :py:class:`ValueError` if collection is not ascending sorted
    Examples:
    >>> __assert_sorted([0, 1, 2, 4])
    True
    >>> __assert_sorted([10, -1, 5])
    Traceback (most recent call last):
        ...
    ValueError: Collection must be ascending sorted
    """
    if collection != sorted(collection):
        raise ValueError("Collection must be ascending sorted")
    return True


if __name__ == "__main__":
    import sys

    """
        user_input = input('Enter numbers separated by comma:\n').strip()
    collection = [int(item) for item in user_input.split(',')]
    try:
        __assert_sorted(collection)
    except ValueError:
        sys.exit('Sequence must be ascending sorted to apply interpolation search')

    target_input = input('Enter a single number to be found in the list:\n')
    target = int(target_input)
        """

    debug = 0
    if debug == 1:
        collection = [10, 30, 40, 45, 50, 66, 77, 93]
        try:
            __assert_sorted(collection)
        except ValueError:
            sys.exit("Sequence must be ascending sorted to apply interpolation search")
        target = 67

    result = interpolation_search(collection, target)
    if result is not None:
        print(f"{target} found at positions: {result}")
    else:
        print("Not found")

## Jump Search
"""
Pure Python implementation of the jump search algorithm.
This algorithm iterates through a sorted collection with a step of n^(1/2),
until the element compared is bigger than the one searched.
It will then perform a linear search until it matches the wanted number.
If not found, it returns -1.
"""

import math


def jump_search(arr: list, x: int) -> int:
    """
    Pure Python implementation of the jump search algorithm.
    Examples:
    >>> jump_search([0, 1, 2, 3, 4, 5], 3)
    3
    >>> jump_search([-5, -2, -1], -1)
    2
    >>> jump_search([0, 5, 10, 20], 8)
    -1
    >>> jump_search([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610], 55)
    10
    """

    n = len(arr)
    step = int(math.floor(math.sqrt(n)))
    prev = 0
    while arr[min(step, n) - 1] < x:
        prev = step
        step += int(math.floor(math.sqrt(n)))
        if prev >= n:
            return -1

    while arr[prev] < x:
        prev = prev + 1
        if prev == min(step, n):
            return -1
    if arr[prev] == x:
        return prev
    return -1


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    arr = [int(item) for item in user_input.split(",")]
    x = int(input("Enter the number to be searched:\n"))
    res = jump_search(arr, x)
    if res == -1:
        print("Number not found!")
    else:
        print(f"Number {x} is at index {res}")

## Linear Search
"""
This is pure Python implementation of linear search algorithm

For doctests run following command:
python3 -m doctest -v linear_search.py

For manual testing run:
python3 linear_search.py
"""


def linear_search(sequence: list, target: int) -> int:
    """A pure Python implementation of a linear search algorithm

    :param sequence: a collection with comparable items (as sorted items not required
        in Linear Search)
    :param target: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> linear_search([0, 5, 7, 10, 15], 0)
    0
    >>> linear_search([0, 5, 7, 10, 15], 15)
    4
    >>> linear_search([0, 5, 7, 10, 15], 5)
    1
    >>> linear_search([0, 5, 7, 10, 15], 6)
    -1
    """
    for index, item in enumerate(sequence):
        if item == target:
            return index
    return -1


def rec_linear_search(sequence: list, low: int, high: int, target: int) -> int:
    """
    A pure Python implementation of a recursive linear search algorithm

    :param sequence: a collection with comparable items (as sorted items not required
        in Linear Search)
    :param low: Lower bound of the array
    :param high: Higher bound of the array
    :param target: The element to be found
    :return: Index of the key or -1 if key not found

    Examples:
    >>> rec_linear_search([0, 30, 500, 100, 700], 0, 4, 0)
    0
    >>> rec_linear_search([0, 30, 500, 100, 700], 0, 4, 700)
    4
    >>> rec_linear_search([0, 30, 500, 100, 700], 0, 4, 30)
    1
    >>> rec_linear_search([0, 30, 500, 100, 700], 0, 4, -6)
    -1
    """
    if not (0 <= high < len(sequence) and 0 <= low < len(sequence)):
        raise Exception("Invalid upper or lower bound!")
    if high < low:
        return -1
    if sequence[low] == target:
        return low
    if sequence[high] == target:
        return high
    return rec_linear_search(sequence, low + 1, high - 1, target)


if __name__ == "__main__":
    user_input = input("Enter numbers separated by comma:\n").strip()
    sequence = [int(item.strip()) for item in user_input.split(",")]

    target = int(input("Enter a single number to be found in the list:\n").strip())
    result = linear_search(sequence, target)
    if result != -1:
        print(f"linear_search({sequence}, {target}) = {result}")
    else:
        print(f"{target} was not found in {sequence}")

## Quick Select
"""
A Python implementation of the quick select algorithm, which is efficient for
calculating the value that would appear in the index of a list if it would be
sorted, even if it is not already sorted
https://en.wikipedia.org/wiki/Quickselect
"""
import random


def _partition(data: list, pivot) -> tuple:
    """
    Three way partition the data into smaller, equal and greater lists,
    in relationship to the pivot
    :param data: The data to be sorted (a list)
    :param pivot: The value to partition the data on
    :return: Three list: smaller, equal and greater
    """
    less, equal, greater = [], [], []
    for element in data:
        if element < pivot:
            less.append(element)
        elif element > pivot:
            greater.append(element)
        else:
            equal.append(element)
    return less, equal, greater


def quick_select(items: list, index: int):
    """
    >>> quick_select([2, 4, 5, 7, 899, 54, 32], 5)
    54
    >>> quick_select([2, 4, 5, 7, 899, 54, 32], 1)
    4
    >>> quick_select([5, 4, 3, 2], 2)
    4
    >>> quick_select([3, 5, 7, 10, 2, 12], 3)
    7
    """
    # index = len(items) // 2 when trying to find the median
    #   (value of index when items is sorted)

    # invalid input
    if index >= len(items) or index < 0:
        return None

    pivot = items[random.randint(0, len(items) - 1)]
    count = 0
    smaller, equal, larger = _partition(items, pivot)
    count = len(equal)
    m = len(smaller)

    # index is the pivot
    if m <= index < m + count:
        return pivot
    # must be in smaller
    elif m > index:
        return quick_select(smaller, index)
    # must be in larger
    else:
        return quick_select(larger, index - (m + count))
    
## Sentinel Linear Search
"""
This is pure Python implementation of sentinel linear search algorithm

For doctests run following command:
python -m doctest -v sentinel_linear_search.py
or
python3 -m doctest -v sentinel_linear_search.py

For manual testing run:
python sentinel_linear_search.py
"""


def sentinel_linear_search(sequence, target):
    """Pure implementation of sentinel linear search algorithm in Python

    :param sequence: some sequence with comparable items
    :param target: item value to search
    :return: index of found item or None if item is not found

    Examples:
    >>> sentinel_linear_search([0, 5, 7, 10, 15], 0)
    0

    >>> sentinel_linear_search([0, 5, 7, 10, 15], 15)
    4

    >>> sentinel_linear_search([0, 5, 7, 10, 15], 5)
    1

    >>> sentinel_linear_search([0, 5, 7, 10, 15], 6)

    """
    sequence.append(target)

    index = 0
    while sequence[index] != target:
        index += 1

    sequence.pop()

    if index == len(sequence):
        return None

    return index


if __name__ == "__main__":
    user_input = input("Enter numbers separated by comma:\n").strip()
    sequence = [int(item) for item in user_input.split(",")]

    target_input = input("Enter a single number to be found in the list:\n")
    target = int(target_input)
    result = sentinel_linear_search(sequence, target)
    if result is not None:
        print(f"{target} found at positions: {result}")
    else:
        print("Not found")


## Simple Binary Search
"""
Pure Python implementation of a binary search algorithm.

For doctests run following command:
python3 -m doctest -v simple_binary_search.py

For manual testing run:
python3 simple_binary_search.py
"""
#from __future__ import annotations


def binary_search(a_list: list[int], item: int) -> bool:
    """
    >>> test_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    >>> binary_search(test_list, 3)
    False
    >>> binary_search(test_list, 13)
    True
    >>> binary_search([4, 4, 5, 6, 7], 4)
    True
    >>> binary_search([4, 4, 5, 6, 7], -10)
    False
    >>> binary_search([-18, 2], -18)
    True
    >>> binary_search([5], 5)
    True
    >>> binary_search(['a', 'c', 'd'], 'c')
    True
    >>> binary_search(['a', 'c', 'd'], 'f')
    False
    >>> binary_search([], 1)
    False
    >>> binary_search([-.1, .1 , .8], .1)
    True
    >>> binary_search(range(-5000, 5000, 10), 80)
    True
    >>> binary_search(range(-5000, 5000, 10), 1255)
    False
    >>> binary_search(range(0, 10000, 5), 2)
    False
    """
    if len(a_list) == 0:
        return False
    midpoint = len(a_list) // 2
    if a_list[midpoint] == item:
        return True
    if item < a_list[midpoint]:
        return binary_search(a_list[:midpoint], item)
    else:
        return binary_search(a_list[midpoint + 1 :], item)


if __name__ == "__main__":
    user_input = input("Enter numbers separated by comma:\n").strip()
    sequence = [int(item.strip()) for item in user_input.split(",")]
    target = int(input("Enter the number to be found in the list:\n").strip())
    not_str = "" if binary_search(sequence, target) else "not "
    print(f"{target} was {not_str}found in {sequence}")

## Simulated Annealing
# https://en.wikipedia.org/wiki/Simulated_annealing
import math
import random
from typing import Any

#from .hill_climbing import SearchProblem


def simulated_annealing(
    search_prob,
    find_max: bool = True,
    max_x: float = math.inf,
    min_x: float = -math.inf,
    max_y: float = math.inf,
    min_y: float = -math.inf,
    visualization: bool = False,
    start_temperate: float = 100,
    rate_of_decrease: float = 0.01,
    threshold_temp: float = 1,
) -> Any:
    """
    Implementation of the simulated annealing algorithm. We start with a given state,
    find all its neighbors. Pick a random neighbor, if that neighbor improves the
    solution, we move in that direction, if that neighbor does not improve the solution,
    we generate a random real number between 0 and 1, if the number is within a certain
    range (calculated using temperature) we move in that direction, else we pick
    another neighbor randomly and repeat the process.

    Args:
        search_prob: The search state at the start.
        find_max: If True, the algorithm should find the minimum else the minimum.
        max_x, min_x, max_y, min_y: the maximum and minimum bounds of x and y.
        visualization: If True, a matplotlib graph is displayed.
        start_temperate: the initial temperate of the system when the program starts.
        rate_of_decrease: the rate at which the temperate decreases in each iteration.
        threshold_temp: the threshold temperature below which we end the search
    Returns a search state having the maximum (or minimum) score.
    """
    search_end = False
    current_state = search_prob
    current_temp = start_temperate
    scores = []
    iterations = 0
    best_state = None

    while not search_end:
        current_score = current_state.score()
        if best_state is None or current_score > best_state.score():
            best_state = current_state
        scores.append(current_score)
        iterations += 1
        next_state = None
        neighbors = current_state.get_neighbors()
        while (
            next_state is None and neighbors
        ):  # till we do not find a neighbor that we can move to
            index = random.randint(0, len(neighbors) - 1)  # picking a random neighbor
            picked_neighbor = neighbors.pop(index)
            change = picked_neighbor.score() - current_score

            if (
                picked_neighbor.x > max_x
                or picked_neighbor.x < min_x
                or picked_neighbor.y > max_y
                or picked_neighbor.y < min_y
            ):
                continue  # neighbor outside our bounds

            if not find_max:
                change = change * -1  # in case we are finding minimum
            if change > 0:  # improves the solution
                next_state = picked_neighbor
            else:
                probability = (math.e) ** (
                    change / current_temp
                )  # probability generation function
                if random.random() < probability:  # random number within probability
                    next_state = picked_neighbor
        current_temp = current_temp - (current_temp * rate_of_decrease)

        if current_temp < threshold_temp or next_state is None:
            # temperature below threshold, or could not find a suitable neighbor
            search_end = True
        else:
            current_state = next_state

    if visualization:
        from matplotlib import pyplot as plt

        plt.plot(range(iterations), scores)
        plt.xlabel("Iterations")
        plt.ylabel("Function values")
        plt.show()
    return best_state


if __name__ == "__main__":

    def test_f1(x, y):
        return (x**2) + (y**2)

    # starting the problem with initial coordinates (12, 47)
    prob = SearchProblem(x=12, y=47, step_size=1, function_to_optimize=test_f1)
    local_min = simulated_annealing(
        prob, find_max=False, max_x=100, min_x=5, max_y=50, min_y=-5, visualization=True
    )
    print(
        "The minimum score for f(x, y) = x^2 + y^2 with the domain 100 > x > 5 "
        f"and 50 > y > - 5 found via hill climbing: {local_min.score()}"
    )

    # starting the problem with initial coordinates (12, 47)
    prob = SearchProblem(x=12, y=47, step_size=1, function_to_optimize=test_f1)
    local_min = simulated_annealing(
        prob, find_max=True, max_x=100, min_x=5, max_y=50, min_y=-5, visualization=True
    )
    print(
        "The maximum score for f(x, y) = x^2 + y^2 with the domain 100 > x > 5 "
        f"and 50 > y > - 5 found via hill climbing: {local_min.score()}"
    )

    def test_f2(x, y):
        return (3 * x**2) - (6 * y)

    prob = SearchProblem(x=3, y=4, step_size=1, function_to_optimize=test_f1)
    local_min = simulated_annealing(prob, find_max=False, visualization=True)
    print(
        "The minimum score for f(x, y) = 3*x^2 - 6*y found via hill climbing: "
        f"{local_min.score()}"
    )

    prob = SearchProblem(x=3, y=4, step_size=1, function_to_optimize=test_f1)
    local_min = simulated_annealing(prob, find_max=True, visualization=True)
    print(
        "The maximum score for f(x, y) = 3*x^2 - 6*y found via hill climbing: "
        f"{local_min.score()}"
    )

## Tabu Search
"""
This is pure Python implementation of Tabu search algorithm for a Travelling Salesman
Problem, that the distances between the cities are symmetric (the distance between city
'a' and city 'b' is the same between city 'b' and city 'a').
The TSP can be represented into a graph. The cities are represented by nodes and the
distance between them is represented by the weight of the ark between the nodes.

The .txt file with the graph has the form:

node1 node2 distance_between_node1_and_node2
node1 node3 distance_between_node1_and_node3
...

Be careful node1, node2 and the distance between them, must exist only once. This means
in the .txt file should not exist:
node1 node2 distance_between_node1_and_node2
node2 node1 distance_between_node2_and_node1

For pytests run following command:
pytest

For manual testing run:
python tabu_search.py -f your_file_name.txt -number_of_iterations_of_tabu_search \
    -s size_of_tabu_search
e.g. python tabu_search.py -f tabudata2.txt -i 4 -s 3
"""
import argparse
import copy


def generate_neighbours(path):
    """
    Pure implementation of generating a dictionary of neighbors and the cost with each
    neighbor, given a path file that includes a graph.

    :param path: The path to the .txt file that includes the graph (e.g.tabudata2.txt)
    :return dict_of_neighbours: Dictionary with key each node and value a list of lists
        with the neighbors of the node and the cost (distance) for each neighbor.

    Example of dict_of_neighbours:
    >>) dict_of_neighbours[a]
    [[b,20],[c,18],[d,22],[e,26]]

    This indicates the neighbors of node (city) 'a', which has neighbor the node 'b'
    with distance 20, the node 'c' with distance 18, the node 'd' with distance 22 and
    the node 'e' with distance 26.
    """

    dict_of_neighbours = {}

    with open(path) as f:
        for line in f:
            if line.split()[0] not in dict_of_neighbours:
                _list = []
                _list.append([line.split()[1], line.split()[2]])
                dict_of_neighbours[line.split()[0]] = _list
            else:
                dict_of_neighbours[line.split()[0]].append(
                    [line.split()[1], line.split()[2]]
                )
            if line.split()[1] not in dict_of_neighbours:
                _list = []
                _list.append([line.split()[0], line.split()[2]])
                dict_of_neighbours[line.split()[1]] = _list
            else:
                dict_of_neighbours[line.split()[1]].append(
                    [line.split()[0], line.split()[2]]
                )

    return dict_of_neighbours


def generate_first_solution(path, dict_of_neighbours):
    """
    Pure implementation of generating the first solution for the Tabu search to start,
    with the redundant resolution strategy. That means that we start from the starting
    node (e.g. node 'a'), then we go to the city nearest (lowest distance) to this node
    (let's assume is node 'c'), then we go to the nearest city of the node 'c', etc.
    till we have visited all cities and return to the starting node.

    :param path: The path to the .txt file that includes the graph (e.g.tabudata2.txt)
    :param dict_of_neighbours: Dictionary with key each node and value a list of lists
        with the neighbors of the node and the cost (distance) for each neighbor.
    :return first_solution: The solution for the first iteration of Tabu search using
        the redundant resolution strategy in a list.
    :return distance_of_first_solution: The total distance that Travelling Salesman
        will travel, if he follows the path in first_solution.
    """

    with open(path) as f:
        start_node = f.read(1)
    end_node = start_node

    first_solution = []

    visiting = start_node

    distance_of_first_solution = 0
    while visiting not in first_solution:
        minim = 10000
        for k in dict_of_neighbours[visiting]:
            if int(k[1]) < int(minim) and k[0] not in first_solution:
                minim = k[1]
                best_node = k[0]

        first_solution.append(visiting)
        distance_of_first_solution = distance_of_first_solution + int(minim)
        visiting = best_node

    first_solution.append(end_node)

    position = 0
    for k in dict_of_neighbours[first_solution[-2]]:
        if k[0] == start_node:
            break
        position += 1

    distance_of_first_solution = (
        distance_of_first_solution
        + int(dict_of_neighbours[first_solution[-2]][position][1])
        - 10000
    )
    return first_solution, distance_of_first_solution


def find_neighborhood(solution, dict_of_neighbours):
    """
    Pure implementation of generating the neighborhood (sorted by total distance of
    each solution from lowest to highest) of a solution with 1-1 exchange method, that
    means we exchange each node in a solution with each other node and generating a
    number of solution named neighborhood.

    :param solution: The solution in which we want to find the neighborhood.
    :param dict_of_neighbours: Dictionary with key each node and value a list of lists
        with the neighbors of the node and the cost (distance) for each neighbor.
    :return neighborhood_of_solution: A list that includes the solutions and the total
        distance of each solution (in form of list) that are produced with 1-1 exchange
        from the solution that the method took as an input

    Example:
    >>> find_neighborhood(['a', 'c', 'b', 'd', 'e', 'a'],
    ...                   {'a': [['b', '20'], ['c', '18'], ['d', '22'], ['e', '26']],
    ...                    'c': [['a', '18'], ['b', '10'], ['d', '23'], ['e', '24']],
    ...                    'b': [['a', '20'], ['c', '10'], ['d', '11'], ['e', '12']],
    ...                    'e': [['a', '26'], ['b', '12'], ['c', '24'], ['d', '40']],
    ...                    'd': [['a', '22'], ['b', '11'], ['c', '23'], ['e', '40']]}
    ...                   )  # doctest: +NORMALIZE_WHITESPACE
    [['a', 'e', 'b', 'd', 'c', 'a', 90],
     ['a', 'c', 'd', 'b', 'e', 'a', 90],
     ['a', 'd', 'b', 'c', 'e', 'a', 93],
     ['a', 'c', 'b', 'e', 'd', 'a', 102],
     ['a', 'c', 'e', 'd', 'b', 'a', 113],
     ['a', 'b', 'c', 'd', 'e', 'a', 119]]
    """

    neighborhood_of_solution = []

    for n in solution[1:-1]:
        idx1 = solution.index(n)
        for kn in solution[1:-1]:
            idx2 = solution.index(kn)
            if n == kn:
                continue

            _tmp = copy.deepcopy(solution)
            _tmp[idx1] = kn
            _tmp[idx2] = n

            distance = 0

            for k in _tmp[:-1]:
                next_node = _tmp[_tmp.index(k) + 1]
                for i in dict_of_neighbours[k]:
                    if i[0] == next_node:
                        distance = distance + int(i[1])
            _tmp.append(distance)

            if _tmp not in neighborhood_of_solution:
                neighborhood_of_solution.append(_tmp)

    index_of_last_item_in_the_list = len(neighborhood_of_solution[0]) - 1

    neighborhood_of_solution.sort(key=lambda x: x[index_of_last_item_in_the_list])
    return neighborhood_of_solution


def tabu_search(
    first_solution, distance_of_first_solution, dict_of_neighbours, iters, size
):
    """
    Pure implementation of Tabu search algorithm for a Travelling Salesman Problem in
    Python.

    :param first_solution: The solution for the first iteration of Tabu search using
        the redundant resolution strategy in a list.
    :param distance_of_first_solution: The total distance that Travelling Salesman will
        travel, if he follows the path in first_solution.
    :param dict_of_neighbours: Dictionary with key each node and value a list of lists
        with the neighbors of the node and the cost (distance) for each neighbor.
    :param iters: The number of iterations that Tabu search will execute.
    :param size: The size of Tabu List.
    :return best_solution_ever: The solution with the lowest distance that occurred
        during the execution of Tabu search.
    :return best_cost: The total distance that Travelling Salesman will travel, if he
        follows the path in best_solution ever.
    """
    count = 1
    solution = first_solution
    tabu_list = []
    best_cost = distance_of_first_solution
    best_solution_ever = solution

    while count <= iters:
        neighborhood = find_neighborhood(solution, dict_of_neighbours)
        index_of_best_solution = 0
        best_solution = neighborhood[index_of_best_solution]
        best_cost_index = len(best_solution) - 1

        found = False
        while not found:
            i = 0
            while i < len(best_solution):
                if best_solution[i] != solution[i]:
                    first_exchange_node = best_solution[i]
                    second_exchange_node = solution[i]
                    break
                i = i + 1

            if [first_exchange_node, second_exchange_node] not in tabu_list and [
                second_exchange_node,
                first_exchange_node,
            ] not in tabu_list:
                tabu_list.append([first_exchange_node, second_exchange_node])
                found = True
                solution = best_solution[:-1]
                cost = neighborhood[index_of_best_solution][best_cost_index]
                if cost < best_cost:
                    best_cost = cost
                    best_solution_ever = solution
            else:
                index_of_best_solution = index_of_best_solution + 1
                best_solution = neighborhood[index_of_best_solution]

        if len(tabu_list) >= size:
            tabu_list.pop(0)

        count = count + 1

    return best_solution_ever, best_cost


def main(args=None):
    dict_of_neighbours = generate_neighbours(args.File)

    first_solution, distance_of_first_solution = generate_first_solution(
        args.File, dict_of_neighbours
    )

    best_sol, best_cost = tabu_search(
        first_solution,
        distance_of_first_solution,
        dict_of_neighbours,
        args.Iterations,
        args.Size,
    )

    print(f"Best solution: {best_sol}, with total distance: {best_cost}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabu Search")
    parser.add_argument(
        "-f",
        "--File",
        type=str,
        help="Path to the file containing the data",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--Iterations",
        type=int,
        help="How many iterations the algorithm should perform",
        required=True,
    )
    parser.add_argument(
        "-s", "--Size", type=int, help="Size of the tabu list", required=True
    )

    # Pass the arguments to main method
    main(parser.parse_args())

## Ternary Search
"""
This is a type of divide and conquer algorithm which divides the search space into
3 parts and finds the target value based on the property of the array or list
(usually monotonic property).

Time Complexity  : O(log3 N)
Space Complexity : O(1)
"""
#from __future__ import annotations

# This is the precision for this function which can be altered.
# It is recommended for users to keep this number greater than or equal to 10.
precision = 10


# This is the linear search that will occur after the search space has become smaller.


def lin_search(left: int, right: int, array: list[int], target: int) -> int:
    """Perform linear search in list. Returns -1 if element is not found.

    Parameters
    ----------
    left : int
        left index bound.
    right : int
        right index bound.
    array : List[int]
        List of elements to be searched on
    target : int
        Element that is searched

    Returns
    -------
    int
        index of element that is looked for.

    Examples
    --------
    >>> lin_search(0, 4, [4, 5, 6, 7], 7)
    3
    >>> lin_search(0, 3, [4, 5, 6, 7], 7)
    -1
    >>> lin_search(0, 2, [-18, 2], -18)
    0
    >>> lin_search(0, 1, [5], 5)
    0
    >>> lin_search(0, 3, ['a', 'c', 'd'], 'c')
    1
    >>> lin_search(0, 3, [.1, .4 , -.1], .1)
    0
    >>> lin_search(0, 3, [.1, .4 , -.1], -.1)
    2
    """
    for i in range(left, right):
        if array[i] == target:
            return i
    return -1


def ite_ternary_search(array: list[int], target: int) -> int:
    """Iterative method of the ternary search algorithm.
    >>> test_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    >>> ite_ternary_search(test_list, 3)
    -1
    >>> ite_ternary_search(test_list, 13)
    4
    >>> ite_ternary_search([4, 5, 6, 7], 4)
    0
    >>> ite_ternary_search([4, 5, 6, 7], -10)
    -1
    >>> ite_ternary_search([-18, 2], -18)
    0
    >>> ite_ternary_search([5], 5)
    0
    >>> ite_ternary_search(['a', 'c', 'd'], 'c')
    1
    >>> ite_ternary_search(['a', 'c', 'd'], 'f')
    -1
    >>> ite_ternary_search([], 1)
    -1
    >>> ite_ternary_search([.1, .4 , -.1], .1)
    0
    """

    left = 0
    right = len(array)
    while left <= right:
        if right - left < precision:
            return lin_search(left, right, array, target)

        one_third = (left + right) // 3 + 1
        two_third = 2 * (left + right) // 3 + 1

        if array[one_third] == target:
            return one_third
        elif array[two_third] == target:
            return two_third

        elif target < array[one_third]:
            right = one_third - 1
        elif array[two_third] < target:
            left = two_third + 1

        else:
            left = one_third + 1
            right = two_third - 1
    else:
        return -1


def rec_ternary_search(left: int, right: int, array: list[int], target: int) -> int:
    """Recursive method of the ternary search algorithm.

    >>> test_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    >>> rec_ternary_search(0, len(test_list), test_list, 3)
    -1
    >>> rec_ternary_search(4, len(test_list), test_list, 42)
    8
    >>> rec_ternary_search(0, 2, [4, 5, 6, 7], 4)
    0
    >>> rec_ternary_search(0, 3, [4, 5, 6, 7], -10)
    -1
    >>> rec_ternary_search(0, 1, [-18, 2], -18)
    0
    >>> rec_ternary_search(0, 1, [5], 5)
    0
    >>> rec_ternary_search(0, 2, ['a', 'c', 'd'], 'c')
    1
    >>> rec_ternary_search(0, 2, ['a', 'c', 'd'], 'f')
    -1
    >>> rec_ternary_search(0, 0, [], 1)
    -1
    >>> rec_ternary_search(0, 3, [.1, .4 , -.1], .1)
    0
    """
    if left < right:
        if right - left < precision:
            return lin_search(left, right, array, target)
        one_third = (left + right) // 3 + 1
        two_third = 2 * (left + right) // 3 + 1

        if array[one_third] == target:
            return one_third
        elif array[two_third] == target:
            return two_third

        elif target < array[one_third]:
            return rec_ternary_search(left, one_third - 1, array, target)
        elif array[two_third] < target:
            return rec_ternary_search(two_third + 1, right, array, target)
        else:
            return rec_ternary_search(one_third + 1, two_third - 1, array, target)
    else:
        return -1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    user_input = input("Enter numbers separated by comma:\n").strip()
    collection = [int(item.strip()) for item in user_input.split(",")]
    assert collection == sorted(collection), f"List must be ordered.\n{collection}."
    target = int(input("Enter the number to be found in the list:\n").strip())
    result1 = ite_ternary_search(collection, target)
    result2 = rec_ternary_search(0, len(collection) - 1, collection, target)
    if result2 != -1:
        print(f"Iterative search: {target} found at positions: {result1}")
        print(f"Recursive search: {target} found at positions: {result2}")
    else:
        print("Not found")

### *** Sorts
## Bead Sort
"""
Bead sort only works for sequences of non-negative integers.
https://en.wikipedia.org/wiki/Bead_sort
"""


def bead_sort(sequence: list) -> list:
    """
    >>> bead_sort([6, 11, 12, 4, 1, 5])
    [1, 4, 5, 6, 11, 12]

    >>> bead_sort([9, 8, 7, 6, 5, 4 ,3, 2, 1])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> bead_sort([5, 0, 4, 3])
    [0, 3, 4, 5]

    >>> bead_sort([8, 2, 1])
    [1, 2, 8]

    >>> bead_sort([1, .9, 0.0, 0, -1, -.9])
    Traceback (most recent call last):
        ...
    TypeError: Sequence must be list of non-negative integers

    >>> bead_sort("Hello world")
    Traceback (most recent call last):
        ...
    TypeError: Sequence must be list of non-negative integers
    """
    if any(not isinstance(x, int) or x < 0 for x in sequence):
        raise TypeError("Sequence must be list of non-negative integers")
    for _ in range(len(sequence)):
        for i, (rod_upper, rod_lower) in enumerate(zip(sequence, sequence[1:])):
            if rod_upper > rod_lower:
                sequence[i] -= rod_upper - rod_lower
                sequence[i + 1] += rod_upper - rod_lower
    return sequence


if __name__ == "__main__":
    assert bead_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert bead_sort([7, 9, 4, 3, 5]) == [3, 4, 5, 7, 9]

## Bitonic Sort
"""
Python program for Bitonic Sort.

Note that this program works only when size of input is a power of 2.
"""
#from __future__ import annotations


def comp_and_swap(array: list[int], index1: int, index2: int, direction: int) -> None:
    """Compare the value at given index1 and index2 of the array and swap them as per
    the given direction.

    The parameter direction indicates the sorting direction, ASCENDING(1) or
    DESCENDING(0); if (a[i] > a[j]) agrees with the direction, then a[i] and a[j] are
    interchanged.

    >>> arr = [12, 42, -21, 1]
    >>> comp_and_swap(arr, 1, 2, 1)
    >>> arr
    [12, -21, 42, 1]

    >>> comp_and_swap(arr, 1, 2, 0)
    >>> arr
    [12, 42, -21, 1]

    >>> comp_and_swap(arr, 0, 3, 1)
    >>> arr
    [1, 42, -21, 12]

    >>> comp_and_swap(arr, 0, 3, 0)
    >>> arr
    [12, 42, -21, 1]
    """
    if (direction == 1 and array[index1] > array[index2]) or (
        direction == 0 and array[index1] < array[index2]
    ):
        array[index1], array[index2] = array[index2], array[index1]


def bitonic_merge(array: list[int], low: int, length: int, direction: int) -> None:
    """
    It recursively sorts a bitonic sequence in ascending order, if direction = 1, and in
    descending if direction = 0.
    The sequence to be sorted starts at index position low, the parameter length is the
    number of elements to be sorted.

    >>> arr = [12, 42, -21, 1]
    >>> bitonic_merge(arr, 0, 4, 1)
    >>> arr
    [-21, 1, 12, 42]

    >>> bitonic_merge(arr, 0, 4, 0)
    >>> arr
    [42, 12, 1, -21]
    """
    if length > 1:
        middle = int(length / 2)
        for i in range(low, low + middle):
            comp_and_swap(array, i, i + middle, direction)
        bitonic_merge(array, low, middle, direction)
        bitonic_merge(array, low + middle, middle, direction)


def bitonic_sort(array: list[int], low: int, length: int, direction: int) -> None:
    """
    This function first produces a bitonic sequence by recursively sorting its two
    halves in opposite sorting orders, and then calls bitonic_merge to make them in the
    same order.

    >>> arr = [12, 34, 92, -23, 0, -121, -167, 145]
    >>> bitonic_sort(arr, 0, 8, 1)
    >>> arr
    [-167, -121, -23, 0, 12, 34, 92, 145]

    >>> bitonic_sort(arr, 0, 8, 0)
    >>> arr
    [145, 92, 34, 12, 0, -23, -121, -167]
    """
    if length > 1:
        middle = int(length / 2)
        bitonic_sort(array, low, middle, 1)
        bitonic_sort(array, low + middle, middle, 0)
        bitonic_merge(array, low, length, direction)


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item.strip()) for item in user_input.split(",")]

    bitonic_sort(unsorted, 0, len(unsorted), 1)
    print("\nSorted array in ascending order is: ", end="")
    print(*unsorted, sep=", ")

    bitonic_merge(unsorted, 0, len(unsorted), 0)
    print("Sorted array in descending order is: ", end="")
    print(*unsorted, sep=", ")

## Bogo Sort
"""
This is a pure Python implementation of the bogosort algorithm,
also known as permutation sort, stupid sort, slowsort, shotgun sort, or monkey sort.
Bogosort generates random permutations until it guesses the correct one.

More info on: https://en.wikipedia.org/wiki/Bogosort

For doctests run following command:
python -m doctest -v bogo_sort.py
or
python3 -m doctest -v bogo_sort.py
For manual testing run:
python bogo_sort.py
"""

import random


def bogo_sort(collection):
    """Pure implementation of the bogosort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending
    Examples:
    >>> bogo_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> bogo_sort([])
    []
    >>> bogo_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    def is_sorted(collection):
        for i in range(len(collection) - 1):
            if collection[i] > collection[i + 1]:
                return False
        return True

    while not is_sorted(collection):
        random.shuffle(collection)
    return collection


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(bogo_sort(unsorted))

## Bubble Sort
def bubble_sort(collection):
    """Pure implementation of bubble sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> bubble_sort([0, 5, 2, 3, 2])
    [0, 2, 2, 3, 5]
    >>> bubble_sort([0, 5, 2, 3, 2]) == sorted([0, 5, 2, 3, 2])
    True
    >>> bubble_sort([]) == sorted([])
    True
    >>> bubble_sort([-2, -45, -5]) == sorted([-2, -45, -5])
    True
    >>> bubble_sort([-23, 0, 6, -4, 34]) == sorted([-23, 0, 6, -4, 34])
    True
    >>> bubble_sort(['d', 'a', 'b', 'e', 'c']) == sorted(['d', 'a', 'b', 'e', 'c'])
    True
    >>> import random
    >>> collection = random.sample(range(-50, 50), 100)
    >>> bubble_sort(collection) == sorted(collection)
    True
    >>> import string
    >>> collection = random.choices(string.ascii_letters + string.digits, k=100)
    >>> bubble_sort(collection) == sorted(collection)
    True
    """
    length = len(collection)
    for i in range(length - 1):
        swapped = False
        for j in range(length - 1 - i):
            if collection[j] > collection[j + 1]:
                swapped = True
                collection[j], collection[j + 1] = collection[j + 1], collection[j]
        if not swapped:
            break  # Stop iteration if the collection is sorted.
    return collection


if __name__ == "__main__":
    import doctest
    import time

    doctest.testmod()

    user_input = input("Enter numbers separated by a comma:").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    start = time.process_time()
    print(*bubble_sort(unsorted), sep=",")
    print(f"Processing time: {(time.process_time() - start)%1e9 + 7}")

## Bucket Sort
#!/usr/bin/env python3
"""
Illustrate how to implement bucket sort algorithm.

Author: OMKAR PATHAK
This program will illustrate how to implement bucket sort algorithm

Wikipedia says: Bucket sort, or bin sort, is a sorting algorithm that works
by distributing the elements of an array into a number of buckets.
Each bucket is then sorted individually, either using a different sorting
algorithm, or by recursively applying the bucket sorting algorithm. It is a
distribution sort, and is a cousin of radix sort in the most to least
significant digit flavour.
Bucket sort is a generalization of pigeonhole sort. Bucket sort can be
implemented with comparisons and therefore can also be considered a
comparison sort algorithm. The computational complexity estimates involve the
number of buckets.

Time Complexity of Solution:
Worst case scenario occurs when all the elements are placed in a single bucket.
The overall performance would then be dominated by the algorithm used to sort each
bucket. In this case, O(n log n), because of TimSort

Average Case O(n + (n^2)/k + k), where k is the number of buckets

If k = O(n), time complexity is O(n)

Source: https://en.wikipedia.org/wiki/Bucket_sort
"""
#from __future__ import annotations


def bucket_sort(my_list: list) -> list:
    """
    >>> data = [-1, 2, -5, 0]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [9, 8, 7, 6, -12]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> data = [.4, 1.2, .1, .2, -.9]
    >>> bucket_sort(data) == sorted(data)
    True
    >>> bucket_sort([]) == sorted([])
    True
    >>> import random
    >>> collection = random.sample(range(-50, 50), 50)
    >>> bucket_sort(collection) == sorted(collection)
    True
    """
    if len(my_list) == 0:
        return []
    min_value, max_value = min(my_list), max(my_list)
    bucket_count = int(max_value - min_value) + 1
    buckets: list[list] = [[] for _ in range(bucket_count)]

    for i in my_list:
        buckets[int(i - min_value)].append(i)

    return [v for bucket in buckets for v in sorted(bucket)]


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    assert bucket_sort([4, 5, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert bucket_sort([0, 1, -10, 15, 2, -2]) == [-10, -2, 0, 1, 2, 15]

## Circle Sort
"""
This is a Python implementation of the circle sort algorithm

For doctests run following command:
python3 -m doctest -v circle_sort.py

For manual testing run:
python3 circle_sort.py
"""


def circle_sort(collection: list) -> list:
    """A pure Python implementation of circle sort algorithm

    :param collection: a mutable collection of comparable items in any order
    :return: the same collection in ascending order

    Examples:
    >>> circle_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> circle_sort([])
    []
    >>> circle_sort([-2, 5, 0, -45])
    [-45, -2, 0, 5]
    >>> collections = ([], [0, 5, 3, 2, 2], [-2, 5, 0, -45])
    >>> all(sorted(collection) == circle_sort(collection) for collection in collections)
    True
    """

    if len(collection) < 2:
        return collection

    def circle_sort_util(collection: list, low: int, high: int) -> bool:
        """
        >>> arr = [5,4,3,2,1]
        >>> circle_sort_util(lst, 0, 2)
        True
        >>> arr
        [3, 4, 5, 2, 1]
        """

        swapped = False

        if low == high:
            return swapped

        left = low
        right = high

        while left < right:
            if collection[left] > collection[right]:
                collection[left], collection[right] = (
                    collection[right],
                    collection[left],
                )
                swapped = True

            left += 1
            right -= 1

        if left == right and collection[left] > collection[right + 1]:
            collection[left], collection[right + 1] = (
                collection[right + 1],
                collection[left],
            )

            swapped = True

        mid = low + int((high - low) / 2)
        left_swap = circle_sort_util(collection, low, mid)
        right_swap = circle_sort_util(collection, mid + 1, high)

        return swapped or left_swap or right_swap

    is_not_sorted = True

    while is_not_sorted is True:
        is_not_sorted = circle_sort_util(collection, 0, len(collection) - 1)

    return collection


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(circle_sort(unsorted))

## Cocktail Shaker Sort
""" https://en.wikipedia.org/wiki/Cocktail_shaker_sort """


def cocktail_shaker_sort(unsorted: list) -> list:
    """
    Pure implementation of the cocktail shaker sort algorithm in Python.
    >>> cocktail_shaker_sort([4, 5, 2, 1, 2])
    [1, 2, 2, 4, 5]

    >>> cocktail_shaker_sort([-4, 5, 0, 1, 2, 11])
    [-4, 0, 1, 2, 5, 11]

    >>> cocktail_shaker_sort([0.1, -2.4, 4.4, 2.2])
    [-2.4, 0.1, 2.2, 4.4]

    >>> cocktail_shaker_sort([1, 2, 3, 4, 5])
    [1, 2, 3, 4, 5]

    >>> cocktail_shaker_sort([-4, -5, -24, -7, -11])
    [-24, -11, -7, -5, -4]
    """
    for i in range(len(unsorted) - 1, 0, -1):
        swapped = False

        for j in range(i, 0, -1):
            if unsorted[j] < unsorted[j - 1]:
                unsorted[j], unsorted[j - 1] = unsorted[j - 1], unsorted[j]
                swapped = True

        for j in range(i):
            if unsorted[j] > unsorted[j + 1]:
                unsorted[j], unsorted[j + 1] = unsorted[j + 1], unsorted[j]
                swapped = True

        if not swapped:
            break
    return unsorted


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(f"{cocktail_shaker_sort(unsorted) = }")

## Comb Sort
"""
This is pure Python implementation of comb sort algorithm.
Comb sort is a relatively simple sorting algorithm originally designed by Wlodzimierz
Dobosiewicz in 1980.  It was rediscovered by Stephen Lacey and Richard Box in 1991.
Comb sort improves on bubble sort algorithm.
In bubble sort, distance (or gap) between two compared elements is always one.
Comb sort improvement is that gap can be much more than 1, in order to prevent slowing
down by small values
at the end of a list.

More info on: https://en.wikipedia.org/wiki/Comb_sort

For doctests run following command:
python -m doctest -v comb_sort.py
or
python3 -m doctest -v comb_sort.py

For manual testing run:
python comb_sort.py
"""


def comb_sort(data: list) -> list:
    """Pure implementation of comb sort algorithm in Python
    :param data: mutable collection with comparable items
    :return: the same collection in ascending order
    Examples:
    >>> comb_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> comb_sort([])
    []
    >>> comb_sort([99, 45, -7, 8, 2, 0, -15, 3])
    [-15, -7, 0, 2, 3, 8, 45, 99]
    """
    shrink_factor = 1.3
    gap = len(data)
    completed = False

    while not completed:
        # Update the gap value for a next comb
        gap = int(gap / shrink_factor)
        if gap <= 1:
            completed = True

        index = 0
        while index + gap < len(data):
            if data[index] > data[index + gap]:
                # Swap values
                data[index], data[index + gap] = data[index + gap], data[index]
                completed = False
            index += 1

    return data


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(comb_sort(unsorted))

## Counting Sort
"""
This is pure Python implementation of counting sort algorithm
For doctests run following command:
python -m doctest -v counting_sort.py
or
python3 -m doctest -v counting_sort.py
For manual testing run:
python counting_sort.py
"""


def counting_sort(collection):
    """Pure implementation of counting sort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending
    Examples:
    >>> counting_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> counting_sort([])
    []
    >>> counting_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    # if the collection is empty, returns empty
    if collection == []:
        return []

    # get some information about the collection
    coll_len = len(collection)
    coll_max = max(collection)
    coll_min = min(collection)

    # create the counting array
    counting_arr_length = coll_max + 1 - coll_min
    counting_arr = [0] * counting_arr_length

    # count how much a number appears in the collection
    for number in collection:
        counting_arr[number - coll_min] += 1

    # sum each position with it's predecessors. now, counting_arr[i] tells
    # us how many elements <= i has in the collection
    for i in range(1, counting_arr_length):
        counting_arr[i] = counting_arr[i] + counting_arr[i - 1]

    # create the output collection
    ordered = [0] * coll_len

    # place the elements in the output, respecting the original order (stable
    # sort) from end to begin, updating counting_arr
    for i in reversed(range(0, coll_len)):
        ordered[counting_arr[collection[i] - coll_min] - 1] = collection[i]
        counting_arr[collection[i] - coll_min] -= 1

    return ordered


def counting_sort_string(string):
    """
    >>> counting_sort_string("thisisthestring")
    'eghhiiinrsssttt'
    """
    return "".join([chr(i) for i in counting_sort([ord(c) for c in string])])


if __name__ == "__main__":
    # Test string sort
    assert counting_sort_string("thisisthestring") == "eghhiiinrsssttt"

    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(counting_sort(unsorted))

## Cycle Sort
"""
Code contributed by Honey Sharma
Source: https://en.wikipedia.org/wiki/Cycle_sort
"""


def cycle_sort(array: list) -> list:
    """
    >>> cycle_sort([4, 3, 2, 1])
    [1, 2, 3, 4]

    >>> cycle_sort([-4, 20, 0, -50, 100, -1])
    [-50, -4, -1, 0, 20, 100]

    >>> cycle_sort([-.1, -.2, 1.3, -.8])
    [-0.8, -0.2, -0.1, 1.3]

    >>> cycle_sort([])
    []
    """
    array_len = len(array)
    for cycle_start in range(0, array_len - 1):
        item = array[cycle_start]

        pos = cycle_start
        for i in range(cycle_start + 1, array_len):
            if array[i] < item:
                pos += 1

        if pos == cycle_start:
            continue

        while item == array[pos]:
            pos += 1

        array[pos], item = item, array[pos]
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, array_len):
                if array[i] < item:
                    pos += 1

            while item == array[pos]:
                pos += 1

            array[pos], item = item, array[pos]

    return array


if __name__ == "__main__":
    assert cycle_sort([4, 5, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert cycle_sort([0, 1, -10, 15, 2, -2]) == [-10, -2, 0, 1, 2, 15]

## Double Sort
def double_sort(lst):
    """This sorting algorithm sorts an array using the principle of bubble sort,
    but does it both from left to right and right to left.
    Hence, it's called "Double sort"
    :param collection: mutable ordered sequence of elements
    :return: the same collection in ascending order
    Examples:
    >>> double_sort([-1 ,-2 ,-3 ,-4 ,-5 ,-6 ,-7])
    [-7, -6, -5, -4, -3, -2, -1]
    >>> double_sort([])
    []
    >>> double_sort([-1 ,-2 ,-3 ,-4 ,-5 ,-6])
    [-6, -5, -4, -3, -2, -1]
    >>> double_sort([-3, 10, 16, -42, 29]) == sorted([-3, 10, 16, -42, 29])
    True
    """
    no_of_elements = len(lst)
    for _ in range(
        0, int(((no_of_elements - 1) / 2) + 1)
    ):  # we don't need to traverse to end of list as
        for j in range(0, no_of_elements - 1):
            if (
                lst[j + 1] < lst[j]
            ):  # applying bubble sort algorithm from left to right (or forwards)
                temp = lst[j + 1]
                lst[j + 1] = lst[j]
                lst[j] = temp
            if (
                lst[no_of_elements - 1 - j] < lst[no_of_elements - 2 - j]
            ):  # applying bubble sort algorithm from right to left (or backwards)
                temp = lst[no_of_elements - 1 - j]
                lst[no_of_elements - 1 - j] = lst[no_of_elements - 2 - j]
                lst[no_of_elements - 2 - j] = temp
    return lst


if __name__ == "__main__":
    print("enter the list to be sorted")
    lst = [int(x) for x in input().split()]  # inputing elements of the list in one line
    sorted_lst = double_sort(lst)
    print("the sorted list is")
    print(sorted_lst)

## Dutch National Flag Sort
"""
A pure implementation of Dutch national flag (DNF) sort algorithm in Python.
Dutch National Flag algorithm is an algorithm originally designed by Edsger Dijkstra.
It is the most optimal sort for 3 unique values (eg. 0, 1, 2) in a sequence.  DNF can
sort a sequence of n size with [0 <= a[i] <= 2] at guaranteed O(n) complexity in a
single pass.

The flag of the Netherlands consists of three colors: white, red, and blue.
The task is to randomly arrange balls of white, red, and blue in such a way that balls
of the same color are placed together.  DNF sorts a sequence of 0, 1, and 2's in linear
time that does not consume any extra space.  This algorithm can be implemented only on
a sequence that contains three unique elements.

1) Time complexity is O(n).
2) Space complexity is O(1).

More info on: https://en.wikipedia.org/wiki/Dutch_national_flag_problem

For doctests run following command:
python3 -m doctest -v dutch_national_flag_sort.py

For manual testing run:
python dnf_sort.py
"""


# Python program to sort a sequence containing only 0, 1 and 2 in a single pass.
red = 0  # The first color of the flag.
white = 1  # The second color of the flag.
blue = 2  # The third color of the flag.
colors = (red, white, blue)


def dutch_national_flag_sort(sequence: list) -> list:
    """
    A pure Python implementation of Dutch National Flag sort algorithm.
    :param data: 3 unique integer values (e.g., 0, 1, 2) in an sequence
    :return: The same collection in ascending order

    >>> dutch_national_flag_sort([])
    []
    >>> dutch_national_flag_sort([0])
    [0]
    >>> dutch_national_flag_sort([2, 1, 0, 0, 1, 2])
    [0, 0, 1, 1, 2, 2]
    >>> dutch_national_flag_sort([0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1])
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    >>> dutch_national_flag_sort("abacab")
    Traceback (most recent call last):
      ...
    ValueError: The elements inside the sequence must contains only (0, 1, 2) values
    >>> dutch_national_flag_sort("Abacab")
    Traceback (most recent call last):
      ...
    ValueError: The elements inside the sequence must contains only (0, 1, 2) values
    >>> dutch_national_flag_sort([3, 2, 3, 1, 3, 0, 3])
    Traceback (most recent call last):
      ...
    ValueError: The elements inside the sequence must contains only (0, 1, 2) values
    >>> dutch_national_flag_sort([-1, 2, -1, 1, -1, 0, -1])
    Traceback (most recent call last):
      ...
    ValueError: The elements inside the sequence must contains only (0, 1, 2) values
    >>> dutch_national_flag_sort([1.1, 2, 1.1, 1, 1.1, 0, 1.1])
    Traceback (most recent call last):
      ...
    ValueError: The elements inside the sequence must contains only (0, 1, 2) values
    """
    if not sequence:
        return []
    if len(sequence) == 1:
        return list(sequence)
    low = 0
    high = len(sequence) - 1
    mid = 0
    while mid <= high:
        if sequence[mid] == colors[0]:
            sequence[low], sequence[mid] = sequence[mid], sequence[low]
            low += 1
            mid += 1
        elif sequence[mid] == colors[1]:
            mid += 1
        elif sequence[mid] == colors[2]:
            sequence[mid], sequence[high] = sequence[high], sequence[mid]
            high -= 1
        else:
            raise ValueError(
                f"The elements inside the sequence must contains only {colors} values"
            )
    return sequence


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    user_input = input("Enter numbers separated by commas:\n").strip()
    unsorted = [int(item.strip()) for item in user_input.split(",")]
    print(f"{dutch_national_flag_sort(unsorted)}")

## Exchange Sort
def exchange_sort(numbers: list[int]) -> list[int]:
    """
    Uses exchange sort to sort a list of numbers.
    Source: https://en.wikipedia.org/wiki/Sorting_algorithm#Exchange_sort
    >>> exchange_sort([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]
    >>> exchange_sort([-1, -2, -3])
    [-3, -2, -1]
    >>> exchange_sort([1, 2, 3, 4, 5])
    [1, 2, 3, 4, 5]
    >>> exchange_sort([0, 10, -2, 5, 3])
    [-2, 0, 3, 5, 10]
    >>> exchange_sort([])
    []
    """
    numbers_length = len(numbers)
    for i in range(numbers_length):
        for j in range(i + 1, numbers_length):
            if numbers[j] < numbers[i]:
                numbers[i], numbers[j] = numbers[j], numbers[i]
    return numbers


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(exchange_sort(unsorted))

## External Sort
#!/usr/bin/env python

#
# Sort large text files in a minimum amount of memory
#
import argparse
import os


class FileSplitter:
    BLOCK_FILENAME_FORMAT = "block_{0}.dat"

    def __init__(self, filename):
        self.filename = filename
        self.block_filenames = []

    def write_block(self, data, block_number):
        filename = self.BLOCK_FILENAME_FORMAT.format(block_number)
        with open(filename, "w") as file:
            file.write(data)
        self.block_filenames.append(filename)

    def get_block_filenames(self):
        return self.block_filenames

    def split(self, block_size, sort_key=None):
        i = 0
        with open(self.filename) as file:
            while True:
                lines = file.readlines(block_size)

                if lines == []:
                    break

                if sort_key is None:
                    lines.sort()
                else:
                    lines.sort(key=sort_key)

                self.write_block("".join(lines), i)
                i += 1

    def cleanup(self):
        map(os.remove, self.block_filenames)


class NWayMerge:
    def select(self, choices):
        min_index = -1
        min_str = None

        for i in range(len(choices)):
            if min_str is None or choices[i] < min_str:
                min_index = i

        return min_index


class FilesArray:
    def __init__(self, files):
        self.files = files
        self.empty = set()
        self.num_buffers = len(files)
        self.buffers = {i: None for i in range(self.num_buffers)}

    def get_dict(self):
        return {
            i: self.buffers[i] for i in range(self.num_buffers) if i not in self.empty
        }

    def refresh(self):
        for i in range(self.num_buffers):
            if self.buffers[i] is None and i not in self.empty:
                self.buffers[i] = self.files[i].readline()

                if self.buffers[i] == "":
                    self.empty.add(i)
                    self.files[i].close()

        if len(self.empty) == self.num_buffers:
            return False

        return True

    def unshift(self, index):
        value = self.buffers[index]
        self.buffers[index] = None

        return value


class FileMerger:
    def __init__(self, merge_strategy):
        self.merge_strategy = merge_strategy

    def merge(self, filenames, outfilename, buffer_size):
        buffers = FilesArray(self.get_file_handles(filenames, buffer_size))
        with open(outfilename, "w", buffer_size) as outfile:
            while buffers.refresh():
                min_index = self.merge_strategy.select(buffers.get_dict())
                outfile.write(buffers.unshift(min_index))

    def get_file_handles(self, filenames, buffer_size):
        files = {}

        for i in range(len(filenames)):
            files[i] = open(filenames[i], "r", buffer_size)  # noqa: UP015

        return files


class ExternalSort:
    def __init__(self, block_size):
        self.block_size = block_size

    def sort(self, filename, sort_key=None):
        num_blocks = self.get_number_blocks(filename, self.block_size)
        splitter = FileSplitter(filename)
        splitter.split(self.block_size, sort_key)

        merger = FileMerger(NWayMerge())
        buffer_size = self.block_size / (num_blocks + 1)
        merger.merge(splitter.get_block_filenames(), filename + ".out", buffer_size)

        splitter.cleanup()

    def get_number_blocks(self, filename, block_size):
        return (os.stat(filename).st_size / block_size) + 1


def parse_memory(string):
    if string[-1].lower() == "k":
        return int(string[:-1]) * 1024
    elif string[-1].lower() == "m":
        return int(string[:-1]) * 1024 * 1024
    elif string[-1].lower() == "g":
        return int(string[:-1]) * 1024 * 1024 * 1024
    else:
        return int(string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mem", help="amount of memory to use for sorting", default="100M"
    )
    parser.add_argument(
        "filename", metavar="<filename>", nargs=1, help="name of file to sort"
    )
    args = parser.parse_args()

    sorter = ExternalSort(parse_memory(args.mem))
    sorter.sort(args.filename[0])


if __name__ == "__main__":
    main()

## Gnome Sort
"""
Gnome Sort Algorithm (A.K.A. Stupid Sort)

This algorithm iterates over a list comparing an element with the previous one.
If order is not respected, it swaps element backward until order is respected with
previous element.  It resumes the initial iteration from element new position.

For doctests run following command:
python3 -m doctest -v gnome_sort.py

For manual testing run:
python3 gnome_sort.py
"""


def gnome_sort(lst: list) -> list:
    """
    Pure implementation of the gnome sort algorithm in Python

    Take some mutable ordered collection with heterogeneous comparable items inside as
    arguments, return the same collection ordered by ascending.

    Examples:
    >>> gnome_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> gnome_sort([])
    []

    >>> gnome_sort([-2, -5, -45])
    [-45, -5, -2]

    >>> "".join(gnome_sort(list(set("Gnomes are stupid!"))))
    ' !Gadeimnoprstu'
    """
    if len(lst) <= 1:
        return lst

    i = 1

    while i < len(lst):
        if lst[i - 1] <= lst[i]:
            i += 1
        else:
            lst[i - 1], lst[i] = lst[i], lst[i - 1]
            i -= 1
            if i == 0:
                i = 1

    return lst


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(gnome_sort(unsorted))

## Heap Sort
"""
This is a pure Python implementation of the heap sort algorithm.

For doctests run following command:
python -m doctest -v heap_sort.py
or
python3 -m doctest -v heap_sort.py

For manual testing run:
python heap_sort.py
"""


def heapify(unsorted, index, heap_size):
    largest = index
    left_index = 2 * index + 1
    right_index = 2 * index + 2
    if left_index < heap_size and unsorted[left_index] > unsorted[largest]:
        largest = left_index

    if right_index < heap_size and unsorted[right_index] > unsorted[largest]:
        largest = right_index

    if largest != index:
        unsorted[largest], unsorted[index] = unsorted[index], unsorted[largest]
        heapify(unsorted, largest, heap_size)


def heap_sort(unsorted):
    """
    Pure implementation of the heap sort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> heap_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> heap_sort([])
    []

    >>> heap_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    n = len(unsorted)
    for i in range(n // 2 - 1, -1, -1):
        heapify(unsorted, i, n)
    for i in range(n - 1, 0, -1):
        unsorted[0], unsorted[i] = unsorted[i], unsorted[0]
        heapify(unsorted, 0, i)
    return unsorted


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(heap_sort(unsorted))

## Insertion Sort
"""
A pure Python implementation of the insertion sort algorithm

This algorithm sorts a collection by comparing adjacent elements.
When it finds that order is not respected, it moves the element compared
backward until the order is correct.  It then goes back directly to the
element's initial position resuming forward comparison.

For doctests run following command:
python3 -m doctest -v insertion_sort.py

For manual testing run:
python3 insertion_sort.py
"""


def insertion_sort(collection: list) -> list:
    """A pure Python implementation of the insertion sort algorithm

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> insertion_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> insertion_sort([]) == sorted([])
    True
    >>> insertion_sort([-2, -5, -45]) == sorted([-2, -5, -45])
    True
    >>> insertion_sort(['d', 'a', 'b', 'e', 'c']) == sorted(['d', 'a', 'b', 'e', 'c'])
    True
    >>> import random
    >>> collection = random.sample(range(-50, 50), 100)
    >>> insertion_sort(collection) == sorted(collection)
    True
    >>> import string
    >>> collection = random.choices(string.ascii_letters + string.digits, k=100)
    >>> insertion_sort(collection) == sorted(collection)
    True
    """

    for insert_index, insert_value in enumerate(collection[1:]):
        temp_index = insert_index
        while insert_index >= 0 and insert_value < collection[insert_index]:
            collection[insert_index + 1] = collection[insert_index]
            insert_index -= 1
        if insert_index != temp_index:
            collection[insert_index + 1] = insert_value
    return collection


if __name__ == "__main__":
    from doctest import testmod

    testmod()

    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(f"{insertion_sort(unsorted) = }")

## Intro Sort
"""
Introspective Sort is hybrid sort (Quick Sort + Heap Sort + Insertion Sort)
if the size of the list is under 16, use insertion sort
https://en.wikipedia.org/wiki/Introsort
"""
import math


def insertion_sort(array: list, start: int = 0, end: int = 0) -> list:
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]

    >>> insertion_sort(array, 0, len(array))
    [1, 2, 4, 6, 7, 8, 8, 12, 14, 14, 22, 23, 27, 45, 56, 79]
    """
    end = end or len(array)
    for i in range(start, end):
        temp_index = i
        temp_index_value = array[i]
        while temp_index != start and temp_index_value < array[temp_index - 1]:
            array[temp_index] = array[temp_index - 1]
            temp_index -= 1
        array[temp_index] = temp_index_value
    return array


def heapify(array: list, index: int, heap_size: int) -> None:  # Max Heap
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]

    >>> heapify(array, len(array) // 2 ,len(array))
    """
    largest = index
    left_index = 2 * index + 1  # Left Node
    right_index = 2 * index + 2  # Right Node

    if left_index < heap_size and array[largest] < array[left_index]:
        largest = left_index

    if right_index < heap_size and array[largest] < array[right_index]:
        largest = right_index

    if largest != index:
        array[index], array[largest] = array[largest], array[index]
        heapify(array, largest, heap_size)


def heap_sort(array: list) -> list:
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]

    >>> heap_sort(array)
    [1, 2, 4, 6, 7, 8, 8, 12, 14, 14, 22, 23, 27, 45, 56, 79]
    """
    n = len(array)

    for i in range(n // 2, -1, -1):
        heapify(array, i, n)

    for i in range(n - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        heapify(array, 0, i)

    return array


def median_of_3(
    array: list, first_index: int, middle_index: int, last_index: int
) -> int:
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]

    >>> median_of_3(array, 0, 0 + ((len(array) - 0) // 2) + 1, len(array) - 1)
    12
    """
    if (array[first_index] > array[middle_index]) != (
        array[first_index] > array[last_index]
    ):
        return array[first_index]
    elif (array[middle_index] > array[first_index]) != (
        array[middle_index] > array[last_index]
    ):
        return array[middle_index]
    else:
        return array[last_index]


def partition(array: list, low: int, high: int, pivot: int) -> int:
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]

    >>> partition(array, 0, len(array), 12)
    8
    """
    i = low
    j = high
    while True:
        while array[i] < pivot:
            i += 1
        j -= 1
        while pivot < array[j]:
            j -= 1
        if i >= j:
            return i
        array[i], array[j] = array[j], array[i]
        i += 1


def sort(array: list) -> list:
    """
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> sort([4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12])
    [1, 2, 4, 6, 7, 8, 8, 12, 14, 14, 22, 23, 27, 45, 56, 79]

    >>> sort([-1, -5, -3, -13, -44])
    [-44, -13, -5, -3, -1]

    >>> sort([])
    []

    >>> sort([5])
    [5]

    >>> sort([-3, 0, -7, 6, 23, -34])
    [-34, -7, -3, 0, 6, 23]

    >>> sort([1.7, 1.0, 3.3, 2.1, 0.3 ])
    [0.3, 1.0, 1.7, 2.1, 3.3]

    >>> sort(['d', 'a', 'b', 'e', 'c'])
    ['a', 'b', 'c', 'd', 'e']
    """
    if len(array) == 0:
        return array
    max_depth = 2 * math.ceil(math.log2(len(array)))
    size_threshold = 16
    return intro_sort(array, 0, len(array), size_threshold, max_depth)


def intro_sort(
    array: list, start: int, end: int, size_threshold: int, max_depth: int
) -> list:
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]

    >>> max_depth = 2 * math.ceil(math.log2(len(array)))

    >>> intro_sort(array, 0, len(array), 16, max_depth)
    [1, 2, 4, 6, 7, 8, 8, 12, 14, 14, 22, 23, 27, 45, 56, 79]
    """
    while end - start > size_threshold:
        if max_depth == 0:
            return heap_sort(array)
        max_depth -= 1
        pivot = median_of_3(array, start, start + ((end - start) // 2) + 1, end - 1)
        p = partition(array, start, end, pivot)
        intro_sort(array, p, end, size_threshold, max_depth)
        end = p
    return insertion_sort(array, start, end)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    user_input = input("Enter numbers separated by a comma : ").strip()
    unsorted = [float(item) for item in user_input.split(",")]
    print(sort(unsorted))

## Iterative Merge Sort
"""
Implementation of iterative merge sort in Python
Author: Aman Gupta

For doctests run following command:
python3 -m doctest -v iterative_merge_sort.py

For manual testing run:
python3 iterative_merge_sort.py
"""

#from __future__ import annotations


def merge(input_list: list, low: int, mid: int, high: int) -> list:
    """
    sorting left-half and right-half individually
    then merging them into result
    """
    result = []
    left, right = input_list[low:mid], input_list[mid : high + 1]
    while left and right:
        result.append((left if left[0] <= right[0] else right).pop(0))
    input_list[low : high + 1] = result + left + right
    return input_list


# iteration over the unsorted list
def iter_merge_sort(input_list: list) -> list:
    """
    Return a sorted copy of the input list

    >>> iter_merge_sort([5, 9, 8, 7, 1, 2, 7])
    [1, 2, 5, 7, 7, 8, 9]
    >>> iter_merge_sort([1])
    [1]
    >>> iter_merge_sort([2, 1])
    [1, 2]
    >>> iter_merge_sort([2, 1, 3])
    [1, 2, 3]
    >>> iter_merge_sort([4, 3, 2, 1])
    [1, 2, 3, 4]
    >>> iter_merge_sort([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]
    >>> iter_merge_sort(['c', 'b', 'a'])
    ['a', 'b', 'c']
    >>> iter_merge_sort([0.3, 0.2, 0.1])
    [0.1, 0.2, 0.3]
    >>> iter_merge_sort(['dep', 'dang', 'trai'])
    ['dang', 'dep', 'trai']
    >>> iter_merge_sort([6])
    [6]
    >>> iter_merge_sort([])
    []
    >>> iter_merge_sort([-2, -9, -1, -4])
    [-9, -4, -2, -1]
    >>> iter_merge_sort([1.1, 1, 0.0, -1, -1.1])
    [-1.1, -1, 0.0, 1, 1.1]
    >>> iter_merge_sort(['c', 'b', 'a'])
    ['a', 'b', 'c']
    >>> iter_merge_sort('cba')
    ['a', 'b', 'c']
    """
    if len(input_list) <= 1:
        return input_list
    input_list = list(input_list)

    # iteration for two-way merging
    p = 2
    while p <= len(input_list):
        # getting low, high and middle value for merge-sort of single list
        for i in range(0, len(input_list), p):
            low = i
            high = i + p - 1
            mid = (low + high + 1) // 2
            input_list = merge(input_list, low, mid, high)
        # final merge of last two parts
        if p * 2 >= len(input_list):
            mid = i
            input_list = merge(input_list, 0, mid, len(input_list) - 1)
            break
        p *= 2

    return input_list


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    if user_input == "":
        unsorted = []
    else:
        unsorted = [int(item.strip()) for item in user_input.split(",")]
    print(iter_merge_sort(unsorted))

## Merge Insertion Sort
"""
This is a pure Python implementation of the merge-insertion sort algorithm
Source: https://en.wikipedia.org/wiki/Merge-insertion_sort

For doctests run following command:
python3 -m doctest -v merge_insertion_sort.py
or
python -m doctest -v merge_insertion_sort.py

For manual testing run:
python3 merge_insertion_sort.py
"""

#from __future__ import annotations


def binary_search_insertion(sorted_list, item):
    """
    >>> binary_search_insertion([1, 2, 7, 9, 10], 4)
    [1, 2, 4, 7, 9, 10]
    """
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        middle = (left + right) // 2
        if left == right:
            if sorted_list[middle] < item:
                left = middle + 1
            break
        elif sorted_list[middle] < item:
            left = middle + 1
        else:
            right = middle - 1
    sorted_list.insert(left, item)
    return sorted_list


def merge(left, right):
    """
    >>> merge([[1, 6], [9, 10]], [[2, 3], [4, 5], [7, 8]])
    [[1, 6], [2, 3], [4, 5], [7, 8], [9, 10]]
    """
    result = []
    while left and right:
        if left[0][0] < right[0][0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    return result + left + right


def sortlist_2d(list_2d):
    """
    >>> sortlist_2d([[9, 10], [1, 6], [7, 8], [2, 3], [4, 5]])
    [[1, 6], [2, 3], [4, 5], [7, 8], [9, 10]]
    """
    length = len(list_2d)
    if length <= 1:
        return list_2d
    middle = length // 2
    return merge(sortlist_2d(list_2d[:middle]), sortlist_2d(list_2d[middle:]))


def merge_insertion_sort(collection: list[int]) -> list[int]:
    """Pure implementation of merge-insertion sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> merge_insertion_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> merge_insertion_sort([99])
    [99]

    >>> merge_insertion_sort([-2, -5, -45])
    [-45, -5, -2]

    Testing with all permutations on range(0,5):
    >>> import itertools
    >>> permutations = list(itertools.permutations([0, 1, 2, 3, 4]))
    >>> all(merge_insertion_sort(p) == [0, 1, 2, 3, 4] for p in permutations)
    True
    """

    if len(collection) <= 1:
        return collection

    """
    Group the items into two pairs, and leave one element if there is a last odd item.

    Example: [999, 100, 75, 40, 10000]
                -> [999, 100], [75, 40]. Leave 10000.
    """
    two_paired_list = []
    has_last_odd_item = False
    for i in range(0, len(collection), 2):
        if i == len(collection) - 1:
            has_last_odd_item = True
        else:
            """
            Sort two-pairs in each groups.

            Example: [999, 100], [75, 40]
                        -> [100, 999], [40, 75]
            """
            if collection[i] < collection[i + 1]:
                two_paired_list.append([collection[i], collection[i + 1]])
            else:
                two_paired_list.append([collection[i + 1], collection[i]])

    """
    Sort two_paired_list.

    Example: [100, 999], [40, 75]
                -> [40, 75], [100, 999]
    """
    sorted_list_2d = sortlist_2d(two_paired_list)

    """
    40 < 100 is sure because it has already been sorted.
    Generate the sorted_list of them so that you can avoid unnecessary comparison.

    Example:
           group0 group1
           40     100
           75     999
        ->
           group0 group1
           [40,   100]
           75     999
    """
    result = [i[0] for i in sorted_list_2d]

    """
    100 < 999 is sure because it has already been sorted.
    Put 999 in last of the sorted_list so that you can avoid unnecessary comparison.

    Example:
           group0 group1
           [40,   100]
           75     999
        ->
           group0 group1
           [40,   100,   999]
           75
    """
    result.append(sorted_list_2d[-1][1])

    """
    Insert the last odd item left if there is.

    Example:
           group0 group1
           [40,   100,   999]
           75
        ->
           group0 group1
           [40,   100,   999,   10000]
           75
    """
    if has_last_odd_item:
        pivot = collection[-1]
        result = binary_search_insertion(result, pivot)

    """
    Insert the remaining items.
    In this case, 40 < 75 is sure because it has already been sorted.
    Therefore, you only need to insert 75 into [100, 999, 10000],
    so that you can avoid unnecessary comparison.

    Example:
           group0 group1
           [40,   100,   999,   10000]
            ^ You don't need to compare with this as 40 < 75 is already sure.
           75
        ->
           [40,   75,    100,   999,   10000]
    """
    is_last_odd_item_inserted_before_this_index = False
    for i in range(len(sorted_list_2d) - 1):
        if result[i] == collection[-1] and has_last_odd_item:
            is_last_odd_item_inserted_before_this_index = True
        pivot = sorted_list_2d[i][1]
        # If last_odd_item is inserted before the item's index,
        # you should forward index one more.
        if is_last_odd_item_inserted_before_this_index:
            result = result[: i + 2] + binary_search_insertion(result[i + 2 :], pivot)
        else:
            result = result[: i + 1] + binary_search_insertion(result[i + 1 :], pivot)

    return result


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(merge_insertion_sort(unsorted))

## Merge Sort
"""
This is a pure Python implementation of the merge sort algorithm.

For doctests run following command:
python -m doctest -v merge_sort.py
or
python3 -m doctest -v merge_sort.py
For manual testing run:
python merge_sort.py
"""


def merge_sort(collection: list) -> list:
    """
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending
    Examples:
    >>> merge_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> merge_sort([])
    []
    >>> merge_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    def merge(left: list, right: list) -> list:
        """
        Merge left and right.

        :param left: left collection
        :param right: right collection
        :return: merge result
        """

        def _merge():
            while left and right:
                yield (left if left[0] <= right[0] else right).pop(0)
            yield from left
            yield from right

        return list(_merge())

    if len(collection) <= 1:
        return collection
    mid = len(collection) // 2
    return merge(merge_sort(collection[:mid]), merge_sort(collection[mid:]))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(*merge_sort(unsorted), sep=",")

## Msd Radix Sort
"""
Python implementation of the MSD radix sort algorithm.
It used the binary representation of the integers to sort
them.
https://en.wikipedia.org/wiki/Radix_sort
"""
#from __future__ import annotations


def msd_radix_sort(list_of_ints: list[int]) -> list[int]:
    """
    Implementation of the MSD radix sort algorithm. Only works
    with positive integers
    :param list_of_ints: A list of integers
    :return: Returns the sorted list
    >>> msd_radix_sort([40, 12, 1, 100, 4])
    [1, 4, 12, 40, 100]
    >>> msd_radix_sort([])
    []
    >>> msd_radix_sort([123, 345, 123, 80])
    [80, 123, 123, 345]
    >>> msd_radix_sort([1209, 834598, 1, 540402, 45])
    [1, 45, 1209, 540402, 834598]
    >>> msd_radix_sort([-1, 34, 45])
    Traceback (most recent call last):
        ...
    ValueError: All numbers must be positive
    """
    if not list_of_ints:
        return []

    if min(list_of_ints) < 0:
        raise ValueError("All numbers must be positive")

    most_bits = max(len(bin(x)[2:]) for x in list_of_ints)
    return _msd_radix_sort(list_of_ints, most_bits)


def _msd_radix_sort(list_of_ints: list[int], bit_position: int) -> list[int]:
    """
    Sort the given list based on the bit at bit_position. Numbers with a
    0 at that position will be at the start of the list, numbers with a
    1 at the end.
    :param list_of_ints: A list of integers
    :param bit_position: the position of the bit that gets compared
    :return: Returns a partially sorted list
    >>> _msd_radix_sort([45, 2, 32], 1)
    [2, 32, 45]
    >>> _msd_radix_sort([10, 4, 12], 2)
    [4, 12, 10]
    """
    if bit_position == 0 or len(list_of_ints) in [0, 1]:
        return list_of_ints

    zeros = []
    ones = []
    # Split numbers based on bit at bit_position from the right
    for number in list_of_ints:
        if (number >> (bit_position - 1)) & 1:
            # number has a one at bit bit_position
            ones.append(number)
        else:
            # number has a zero at bit bit_position
            zeros.append(number)

    # recursively split both lists further
    zeros = _msd_radix_sort(zeros, bit_position - 1)
    ones = _msd_radix_sort(ones, bit_position - 1)

    # recombine lists
    res = zeros
    res.extend(ones)

    return res


def msd_radix_sort_inplace(list_of_ints: list[int]):
    """
    Inplace implementation of the MSD radix sort algorithm.
    Sorts based on the binary representation of the integers.
    >>> lst = [1, 345, 23, 89, 0, 3]
    >>> msd_radix_sort_inplace(lst)
    >>> lst == sorted(lst)
    True
    >>> lst = [1, 43, 0, 0, 0, 24, 3, 3]
    >>> msd_radix_sort_inplace(lst)
    >>> lst == sorted(lst)
    True
    >>> lst = []
    >>> msd_radix_sort_inplace(lst)
    >>> lst == []
    True
    >>> lst = [-1, 34, 23, 4, -42]
    >>> msd_radix_sort_inplace(lst)
    Traceback (most recent call last):
        ...
    ValueError: All numbers must be positive
    """

    length = len(list_of_ints)
    if not list_of_ints or length == 1:
        return

    if min(list_of_ints) < 0:
        raise ValueError("All numbers must be positive")

    most_bits = max(len(bin(x)[2:]) for x in list_of_ints)
    _msd_radix_sort_inplace(list_of_ints, most_bits, 0, length)


def _msd_radix_sort_inplace(
    list_of_ints: list[int], bit_position: int, begin_index: int, end_index: int
):
    """
    Sort the given list based on the bit at bit_position. Numbers with a
    0 at that position will be at the start of the list, numbers with a
    1 at the end.
    >>> lst = [45, 2, 32, 24, 534, 2932]
    >>> _msd_radix_sort_inplace(lst, 1, 0, 3)
    >>> lst == [32, 2, 45, 24, 534, 2932]
    True
    >>> lst = [0, 2, 1, 3, 12, 10, 4, 90, 54, 2323, 756]
    >>> _msd_radix_sort_inplace(lst, 2, 4, 7)
    >>> lst == [0, 2, 1, 3, 12, 4, 10, 90, 54, 2323, 756]
    True
    """
    if bit_position == 0 or end_index - begin_index <= 1:
        return

    bit_position -= 1

    i = begin_index
    j = end_index - 1
    while i <= j:
        changed = False
        if not (list_of_ints[i] >> bit_position) & 1:
            # found zero at the beginning
            i += 1
            changed = True
        if (list_of_ints[j] >> bit_position) & 1:
            # found one at the end
            j -= 1
            changed = True

        if changed:
            continue

        list_of_ints[i], list_of_ints[j] = list_of_ints[j], list_of_ints[i]
        j -= 1
        if j != i:
            i += 1

    _msd_radix_sort_inplace(list_of_ints, bit_position, begin_index, i)
    _msd_radix_sort_inplace(list_of_ints, bit_position, i, end_index)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Natural Sort
#from __future__ import annotations

import re


def natural_sort(input_list: list[str]) -> list[str]:
    """
    Sort the given list of strings in the way that humans expect.

    The normal Python sort algorithm sorts lexicographically,
    so you might not get the results that you expect...

    >>> example1 = ['2 ft 7 in', '1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '7 ft 6 in']
    >>> sorted(example1)
    ['1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '2 ft 7 in', '7 ft 6 in']
    >>> # The natural sort algorithm sort based on meaning and not computer code point.
    >>> natural_sort(example1)
    ['1 ft 5 in', '2 ft 7 in', '2 ft 11 in', '7 ft 6 in', '10 ft 2 in']

    >>> example2 = ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> sorted(example2)
    ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> natural_sort(example2)
    ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']
    """

    def alphanum_key(key):
        return [int(s) if s.isdigit() else s.lower() for s in re.split("([0-9]+)", key)]

    return sorted(input_list, key=alphanum_key)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Odd Even Sort
"""
Odd even sort implementation.

https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort
"""


def odd_even_sort(input_list: list) -> list:
    """
    Sort input with odd even sort.

    This algorithm uses the same idea of bubblesort,
    but by first dividing in two phase (odd and even).
    Originally developed for use on parallel processors
    with local interconnections.
    :param collection: mutable ordered sequence of elements
    :return: same collection in ascending order
    Examples:
    >>> odd_even_sort([5 , 4 ,3 ,2 ,1])
    [1, 2, 3, 4, 5]
    >>> odd_even_sort([])
    []
    >>> odd_even_sort([-10 ,-1 ,10 ,2])
    [-10, -1, 2, 10]
    >>> odd_even_sort([1 ,2 ,3 ,4])
    [1, 2, 3, 4]
    """
    is_sorted = False
    while is_sorted is False:  # Until all the indices are traversed keep looping
        is_sorted = True
        for i in range(0, len(input_list) - 1, 2):  # iterating over all even indices
            if input_list[i] > input_list[i + 1]:
                input_list[i], input_list[i + 1] = input_list[i + 1], input_list[i]
                # swapping if elements not in order
                is_sorted = False

        for i in range(1, len(input_list) - 1, 2):  # iterating over all odd indices
            if input_list[i] > input_list[i + 1]:
                input_list[i], input_list[i + 1] = input_list[i + 1], input_list[i]
                # swapping if elements not in order
                is_sorted = False
    return input_list


if __name__ == "__main__":
    print("Enter list to be sorted")
    input_list = [int(x) for x in input().split()]
    # inputing elements of the list in one line
    sorted_list = odd_even_sort(input_list)
    print("The sorted list is")
    print(sorted_list)

## Odd Even Transposition Parallel
"""
This is an implementation of odd-even transposition sort.

It works by performing a series of parallel swaps between odd and even pairs of
variables in the list.

This implementation represents each variable in the list with a process and
each process communicates with its neighboring processes in the list to perform
comparisons.
They are synchronized with locks and message passing but other forms of
synchronization could be used.
"""
from multiprocessing import Lock, Pipe, Process

# lock used to ensure that two processes do not access a pipe at the same time
process_lock = Lock()

"""
The function run by the processes that sorts the list

position = the position in the list the process represents, used to know which
            neighbor we pass our value to
value = the initial value at list[position]
LSend, RSend = the pipes we use to send to our left and right neighbors
LRcv, RRcv = the pipes we use to receive from our left and right neighbors
resultPipe = the pipe used to send results back to main
"""


def oe_process(position, value, l_send, r_send, lr_cv, rr_cv, result_pipe):
    global process_lock

    # we perform n swaps since after n swaps we know we are sorted
    # we *could* stop early if we are sorted already, but it takes as long to
    # find out we are sorted as it does to sort the list with this algorithm
    for i in range(0, 10):
        if (i + position) % 2 == 0 and r_send is not None:
            # send your value to your right neighbor
            process_lock.acquire()
            r_send[1].send(value)
            process_lock.release()

            # receive your right neighbor's value
            process_lock.acquire()
            temp = rr_cv[0].recv()
            process_lock.release()

            # take the lower value since you are on the left
            value = min(value, temp)
        elif (i + position) % 2 != 0 and l_send is not None:
            # send your value to your left neighbor
            process_lock.acquire()
            l_send[1].send(value)
            process_lock.release()

            # receive your left neighbor's value
            process_lock.acquire()
            temp = lr_cv[0].recv()
            process_lock.release()

            # take the higher value since you are on the right
            value = max(value, temp)
    # after all swaps are performed, send the values back to main
    result_pipe[1].send(value)


"""
the function which creates the processes that perform the parallel swaps

arr = the list to be sorted
"""


def odd_even_transposition(arr):
    process_array_ = []
    result_pipe = []
    # initialize the list of pipes where the values will be retrieved
    for _ in arr:
        result_pipe.append(Pipe())
    # creates the processes
    # the first and last process only have one neighbor so they are made outside
    # of the loop
    temp_rs = Pipe()
    temp_rr = Pipe()
    process_array_.append(
        Process(
            target=oe_process,
            args=(0, arr[0], None, temp_rs, None, temp_rr, result_pipe[0]),
        )
    )
    temp_lr = temp_rs
    temp_ls = temp_rr

    for i in range(1, len(arr) - 1):
        temp_rs = Pipe()
        temp_rr = Pipe()
        process_array_.append(
            Process(
                target=oe_process,
                args=(i, arr[i], temp_ls, temp_rs, temp_lr, temp_rr, result_pipe[i]),
            )
        )
        temp_lr = temp_rs
        temp_ls = temp_rr

    process_array_.append(
        Process(
            target=oe_process,
            args=(
                len(arr) - 1,
                arr[len(arr) - 1],
                temp_ls,
                None,
                temp_lr,
                None,
                result_pipe[len(arr) - 1],
            ),
        )
    )

    # start the processes
    for p in process_array_:
        p.start()

    # wait for the processes to end and write their values to the list
    for p in range(0, len(result_pipe)):
        arr[p] = result_pipe[p][0].recv()
        process_array_[p].join()
    return arr


# creates a reverse sorted list and sorts it
def main():
    arr = list(range(10, 0, -1))
    print("Initial List")
    print(*arr)
    arr = odd_even_transposition(arr)
    print("Sorted List\n")
    print(*arr)


if __name__ == "__main__":
    main()

## Odd Even Transposition Single Threaded
"""
Source: https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

This is a non-parallelized implementation of odd-even transposition sort.

Normally the swaps in each set happen simultaneously, without that the algorithm
is no better than bubble sort.
"""


def odd_even_transposition(arr: list) -> list:
    """
    >>> odd_even_transposition([5, 4, 3, 2, 1])
    [1, 2, 3, 4, 5]

    >>> odd_even_transposition([13, 11, 18, 0, -1])
    [-1, 0, 11, 13, 18]

    >>> odd_even_transposition([-.1, 1.1, .1, -2.9])
    [-2.9, -0.1, 0.1, 1.1]
    """
    arr_size = len(arr)
    for _ in range(arr_size):
        for i in range(_ % 2, arr_size - 1, 2):
            if arr[i + 1] < arr[i]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]

    return arr


if __name__ == "__main__":
    arr = list(range(10, 0, -1))
    print(f"Original: {arr}. Sorted: {odd_even_transposition(arr)}")

## Pancake Sort
"""
This is a pure Python implementation of the pancake sort algorithm
For doctests run following command:
python3 -m doctest -v pancake_sort.py
or
python -m doctest -v pancake_sort.py
For manual testing run:
python pancake_sort.py
"""


def pancake_sort(arr):
    """Sort Array with Pancake Sort.
    :param arr: Collection containing comparable items
    :return: Collection ordered in ascending order of items
    Examples:
    >>> pancake_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> pancake_sort([])
    []
    >>> pancake_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    cur = len(arr)
    while cur > 1:
        # Find the maximum number in arr
        mi = arr.index(max(arr[0:cur]))
        # Reverse from 0 to mi
        arr = arr[mi::-1] + arr[mi + 1 : len(arr)]
        # Reverse whole list
        arr = arr[cur - 1 :: -1] + arr[cur : len(arr)]
        cur -= 1
    return arr


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(pancake_sort(unsorted))

## Patience Sort
#from __future__ import annotations

from bisect import bisect_left
from functools import total_ordering
from heapq import merge

"""
A pure Python implementation of the patience sort algorithm

For more information: https://en.wikipedia.org/wiki/Patience_sorting

This algorithm is based on the card game patience

For doctests run following command:
python3 -m doctest -v patience_sort.py

For manual testing run:
python3 patience_sort.py
"""


@total_ordering
class Stack(list):
    def __lt__(self, other):
        return self[-1] < other[-1]

    def __eq__(self, other):
        return self[-1] == other[-1]


def patience_sort(collection: list) -> list:
    """A pure implementation of patience sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending

    Examples:
    >>> patience_sort([1, 9, 5, 21, 17, 6])
    [1, 5, 6, 9, 17, 21]

    >>> patience_sort([])
    []

    >>> patience_sort([-3, -17, -48])
    [-48, -17, -3]
    """
    stacks: list[Stack] = []
    # sort into stacks
    for element in collection:
        new_stacks = Stack([element])
        i = bisect_left(stacks, new_stacks)
        if i != len(stacks):
            stacks[i].append(element)
        else:
            stacks.append(new_stacks)

    # use a heap-based merge to merge stack efficiently
    collection[:] = merge(*(reversed(stack) for stack in stacks))
    return collection


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(patience_sort(unsorted))
  
## Pigeon Sort
"""
    This is an implementation of Pigeon Hole Sort.
    For doctests run following command:

    python3 -m doctest -v pigeon_sort.py
    or
    python -m doctest -v pigeon_sort.py

    For manual testing run:
    python pigeon_sort.py
"""
#from __future__ import annotations


def pigeon_sort(array: list[int]) -> list[int]:
    """
    Implementation of pigeon hole sort algorithm
    :param array: Collection of comparable items
    :return: Collection sorted in ascending order
    >>> pigeon_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> pigeon_sort([])
    []
    >>> pigeon_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    if len(array) == 0:
        return array

    _min, _max = min(array), max(array)

    # Compute the variables
    holes_range = _max - _min + 1
    holes, holes_repeat = [0] * holes_range, [0] * holes_range

    # Make the sorting.
    for i in array:
        index = i - _min
        holes[index] = i
        holes_repeat[index] += 1

    # Makes the array back by replacing the numbers.
    index = 0
    for i in range(holes_range):
        while holes_repeat[i] > 0:
            array[index] = holes[i]
            index += 1
            holes_repeat[i] -= 1

    # Returns the sorted array.
    return array


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    user_input = input("Enter numbers separated by comma:\n")
    unsorted = [int(x) for x in user_input.split(",")]
    print(pigeon_sort(unsorted))

## Pigeonhole Sort
# Python program to implement Pigeonhole Sorting in python

# Algorithm for the pigeonhole sorting


def pigeonhole_sort(a):
    """
    >>> a = [8, 3, 2, 7, 4, 6, 8]
    >>> b = sorted(a)  # a nondestructive sort
    >>> pigeonhole_sort(a)  # a destructive sort
    >>> a == b
    True
    """
    # size of range of values in the list (ie, number of pigeonholes we need)

    min_val = min(a)  # min() finds the minimum value
    max_val = max(a)  # max() finds the maximum value

    size = max_val - min_val + 1  # size is difference of max and min values plus one

    # list of pigeonholes of size equal to the variable size
    holes = [0] * size

    # Populate the pigeonholes.
    for x in a:
        assert isinstance(x, int), "integers only please"
        holes[x - min_val] += 1

    # Putting the elements back into the array in an order.
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            a[i] = count + min_val
            i += 1


def main():
    a = [8, 3, 2, 7, 4, 6, 8]
    pigeonhole_sort(a)
    print("Sorted order is:", " ".join(a))


if __name__ == "__main__":
    main()

## Quick Sort
"""
A pure Python implementation of the quick sort algorithm

For doctests run following command:
python3 -m doctest -v quick_sort.py

For manual testing run:
python3 quick_sort.py
"""
#from __future__ import annotations

from random import randrange


def quick_sort(collection: list) -> list:
    """A pure Python implementation of quick sort algorithm

    :param collection: a mutable collection of comparable items
    :return: the same collection ordered by ascending

    Examples:
    >>> quick_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> quick_sort([])
    []
    >>> quick_sort([-2, 5, 0, -45])
    [-45, -2, 0, 5]
    """
    if len(collection) < 2:
        return collection
    pivot_index = randrange(len(collection))  # Use random element as pivot
    pivot = collection[pivot_index]
    greater: list[int] = []  # All elements greater than pivot
    lesser: list[int] = []  # All elements less than or equal to pivot

    for element in collection[:pivot_index]:
        (greater if element > pivot else lesser).append(element)

    for element in collection[pivot_index + 1 :]:
        (greater if element > pivot else lesser).append(element)

    return [*quick_sort(lesser), pivot, *quick_sort(greater)]


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(quick_sort(unsorted))

## Quick Sort 3 Partition
def quick_sort_3partition(sorting: list, left: int, right: int) -> None:
    if right <= left:
        return
    a = i = left
    b = right
    pivot = sorting[left]
    while i <= b:
        if sorting[i] < pivot:
            sorting[a], sorting[i] = sorting[i], sorting[a]
            a += 1
            i += 1
        elif sorting[i] > pivot:
            sorting[b], sorting[i] = sorting[i], sorting[b]
            b -= 1
        else:
            i += 1
    quick_sort_3partition(sorting, left, a - 1)
    quick_sort_3partition(sorting, b + 1, right)


def quick_sort_lomuto_partition(sorting: list, left: int, right: int) -> None:
    """
    A pure Python implementation of quick sort algorithm(in-place)
    with Lomuto partition scheme:
    https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme

    :param sorting: sort list
    :param left: left endpoint of sorting
    :param right: right endpoint of sorting
    :return: None

    Examples:
    >>> nums1 = [0, 5, 3, 1, 2]
    >>> quick_sort_lomuto_partition(nums1, 0, 4)
    >>> nums1
    [0, 1, 2, 3, 5]
    >>> nums2 = []
    >>> quick_sort_lomuto_partition(nums2, 0, 0)
    >>> nums2
    []
    >>> nums3 = [-2, 5, 0, -4]
    >>> quick_sort_lomuto_partition(nums3, 0, 3)
    >>> nums3
    [-4, -2, 0, 5]
    """
    if left < right:
        pivot_index = lomuto_partition(sorting, left, right)
        quick_sort_lomuto_partition(sorting, left, pivot_index - 1)
        quick_sort_lomuto_partition(sorting, pivot_index + 1, right)


def lomuto_partition(sorting: list, left: int, right: int) -> int:
    """
    Example:
    >>> lomuto_partition([1,5,7,6], 0, 3)
    2
    """
    pivot = sorting[right]
    store_index = left
    for i in range(left, right):
        if sorting[i] < pivot:
            sorting[store_index], sorting[i] = sorting[i], sorting[store_index]
            store_index += 1
    sorting[right], sorting[store_index] = sorting[store_index], sorting[right]
    return store_index


def three_way_radix_quicksort(sorting: list) -> list:
    """
    Three-way radix quicksort:
    https://en.wikipedia.org/wiki/Quicksort#Three-way_radix_quicksort
    First divide the list into three parts.
    Then recursively sort the "less than" and "greater than" partitions.

    >>> three_way_radix_quicksort([])
    []
    >>> three_way_radix_quicksort([1])
    [1]
    >>> three_way_radix_quicksort([-5, -2, 1, -2, 0, 1])
    [-5, -2, -2, 0, 1, 1]
    >>> three_way_radix_quicksort([1, 2, 5, 1, 2, 0, 0, 5, 2, -1])
    [-1, 0, 0, 1, 1, 2, 2, 2, 5, 5]
    """
    if len(sorting) <= 1:
        return sorting
    return (
        three_way_radix_quicksort([i for i in sorting if i < sorting[0]])
        + [i for i in sorting if i == sorting[0]]
        + three_way_radix_quicksort([i for i in sorting if i > sorting[0]])
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    quick_sort_3partition(unsorted, 0, len(unsorted) - 1)
    print(unsorted)

## Radix Sort
"""
This is a pure Python implementation of the radix sort algorithm

Source: https://en.wikipedia.org/wiki/Radix_sort
"""
#from __future__ import annotations

RADIX = 10


def radix_sort(list_of_ints: list[int]) -> list[int]:
    """
    Examples:
    >>> radix_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> radix_sort(list(range(15))) == sorted(range(15))
    True
    >>> radix_sort(list(range(14,-1,-1))) == sorted(range(15))
    True
    >>> radix_sort([1,100,10,1000]) == sorted([1,100,10,1000])
    True
    """
    placement = 1
    max_digit = max(list_of_ints)
    while placement <= max_digit:
        # declare and initialize empty buckets
        buckets: list[list] = [[] for _ in range(RADIX)]
        # split list_of_ints between the buckets
        for i in list_of_ints:
            tmp = int((i / placement) % RADIX)
            buckets[tmp].append(i)
        # put each buckets' contents into list_of_ints
        a = 0
        for b in range(RADIX):
            for i in buckets[b]:
                list_of_ints[a] = i
                a += 1
        # move to next
        placement *= RADIX
    return list_of_ints


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Random Normal Distribution Quicksort
from random import randint
from tempfile import TemporaryFile

import numpy as np


def _in_place_quick_sort(a, start, end):
    count = 0
    if start < end:
        pivot = randint(start, end)
        temp = a[end]
        a[end] = a[pivot]
        a[pivot] = temp

        p, count = _in_place_partition(a, start, end)
        count += _in_place_quick_sort(a, start, p - 1)
        count += _in_place_quick_sort(a, p + 1, end)
    return count


def _in_place_partition(a, start, end):
    count = 0
    pivot = randint(start, end)
    temp = a[end]
    a[end] = a[pivot]
    a[pivot] = temp
    new_pivot_index = start - 1
    for index in range(start, end):
        count += 1
        if a[index] < a[end]:  # check if current val is less than pivot value
            new_pivot_index = new_pivot_index + 1
            temp = a[new_pivot_index]
            a[new_pivot_index] = a[index]
            a[index] = temp

    temp = a[new_pivot_index + 1]
    a[new_pivot_index + 1] = a[end]
    a[end] = temp
    return new_pivot_index + 1, count


outfile = TemporaryFile()
p = 100  # 1000 elements are to be sorted


mu, sigma = 0, 1  # mean and standard deviation
X = np.random.normal(mu, sigma, p)
np.save(outfile, X)
print("The array is")
print(X)


outfile.seek(0)  # using the same array
M = np.load(outfile)
r = len(M) - 1
z = _in_place_quick_sort(M, 0, r)

print(
    "No of Comparisons for 100 elements selected from a standard normal distribution"
    "is :"
)
print(z)

## Random Pivot Quick Sort
"""
Picks the random index as the pivot
"""
import random


def partition(a, left_index, right_index):
    pivot = a[left_index]
    i = left_index + 1
    for j in range(left_index + 1, right_index):
        if a[j] < pivot:
            a[j], a[i] = a[i], a[j]
            i += 1
    a[left_index], a[i - 1] = a[i - 1], a[left_index]
    return i - 1


def quick_sort_random(a, left, right):
    if left < right:
        pivot = random.randint(left, right - 1)
        a[pivot], a[left] = (
            a[left],
            a[pivot],
        )  # switches the pivot with the left most bound
        pivot_index = partition(a, left, right)
        quick_sort_random(
            a, left, pivot_index
        )  # recursive quicksort to the left of the pivot point
        quick_sort_random(
            a, pivot_index + 1, right
        )  # recursive quicksort to the right of the pivot point


def main():
    user_input = input("Enter numbers separated by a comma:\n").strip()
    arr = [int(item) for item in user_input.split(",")]

    quick_sort_random(arr, 0, len(arr))

    print(arr)


if __name__ == "__main__":
    main()

## Recursive Bubble Sort
def bubble_sort(list_data: list, length: int = 0) -> list:
    """
    It is similar is bubble sort but recursive.
    :param list_data: mutable ordered sequence of elements
    :param length: length of list data
    :return: the same list in ascending order

    >>> bubble_sort([0, 5, 2, 3, 2], 5)
    [0, 2, 2, 3, 5]

    >>> bubble_sort([], 0)
    []

    >>> bubble_sort([-2, -45, -5], 3)
    [-45, -5, -2]

    >>> bubble_sort([-23, 0, 6, -4, 34], 5)
    [-23, -4, 0, 6, 34]

    >>> bubble_sort([-23, 0, 6, -4, 34], 5) == sorted([-23, 0, 6, -4, 34])
    True

    >>> bubble_sort(['z','a','y','b','x','c'], 6)
    ['a', 'b', 'c', 'x', 'y', 'z']

    >>> bubble_sort([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6])
    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
    """
    length = length or len(list_data)
    swapped = False
    for i in range(length - 1):
        if list_data[i] > list_data[i + 1]:
            list_data[i], list_data[i + 1] = list_data[i + 1], list_data[i]
            swapped = True

    return list_data if not swapped else bubble_sort(list_data, length - 1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Recursive Insertion Sort
"""
A recursive implementation of the insertion sort algorithm
"""
#from __future__ import annotations


def rec_insertion_sort(collection: list, n: int):
    """
    Given a collection of numbers and its length, sorts the collections
    in ascending order

    :param collection: A mutable collection of comparable elements
    :param n: The length of collections

    >>> col = [1, 2, 1]
    >>> rec_insertion_sort(col, len(col))
    >>> col
    [1, 1, 2]

    >>> col = [2, 1, 0, -1, -2]
    >>> rec_insertion_sort(col, len(col))
    >>> col
    [-2, -1, 0, 1, 2]

    >>> col = [1]
    >>> rec_insertion_sort(col, len(col))
    >>> col
    [1]
    """
    # Checks if the entire collection has been sorted
    if len(collection) <= 1 or n <= 1:
        return

    insert_next(collection, n - 1)
    rec_insertion_sort(collection, n - 1)


def insert_next(collection: list, index: int):
    """
    Inserts the '(index-1)th' element into place

    >>> col = [3, 2, 4, 2]
    >>> insert_next(col, 1)
    >>> col
    [2, 3, 4, 2]

    >>> col = [3, 2, 3]
    >>> insert_next(col, 2)
    >>> col
    [3, 2, 3]

    >>> col = []
    >>> insert_next(col, 1)
    >>> col
    []
    """
    # Checks order between adjacent elements
    if index >= len(collection) or collection[index - 1] <= collection[index]:
        return

    # Swaps adjacent elements since they are not in ascending order
    collection[index - 1], collection[index] = (
        collection[index],
        collection[index - 1],
    )

    insert_next(collection, index + 1)


if __name__ == "__main__":
    numbers = input("Enter integers separated by spaces: ")
    number_list: list[int] = [int(num) for num in numbers.split()]
    rec_insertion_sort(number_list, len(number_list))
    print(number_list)

## Recursive Mergesort Array
"""A merge sort which accepts an array as input and recursively
splits an array in half and sorts and combines them.
"""

"""https://en.wikipedia.org/wiki/Merge_sort """


def merge(arr: list[int]) -> list[int]:
    """Return a sorted array.
    >>> merge([10,9,8,7,6,5,4,3,2,1])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> merge([1,2,3,4,5,6,7,8,9,10])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> merge([10,22,1,2,3,9,15,23])
    [1, 2, 3, 9, 10, 15, 22, 23]
    >>> merge([100])
    [100]
    >>> merge([])
    []
    """
    if len(arr) > 1:
        middle_length = len(arr) // 2  # Finds the middle of the array
        left_array = arr[
            :middle_length
        ]  # Creates an array of the elements in the first half.
        right_array = arr[
            middle_length:
        ]  # Creates an array of the elements in the second half.
        left_size = len(left_array)
        right_size = len(right_array)
        merge(left_array)  # Starts sorting the left.
        merge(right_array)  # Starts sorting the right
        left_index = 0  # Left Counter
        right_index = 0  # Right Counter
        index = 0  # Position Counter
        while (
            left_index < left_size and right_index < right_size
        ):  # Runs until the lowers size of the left and right are sorted.
            if left_array[left_index] < right_array[right_index]:
                arr[index] = left_array[left_index]
                left_index += 1
            else:
                arr[index] = right_array[right_index]
                right_index += 1
            index += 1
        while (
            left_index < left_size
        ):  # Adds the left over elements in the left half of the array
            arr[index] = left_array[left_index]
            left_index += 1
            index += 1
        while (
            right_index < right_size
        ):  # Adds the left over elements in the right half of the array
            arr[index] = right_array[right_index]
            right_index += 1
            index += 1
    return arr


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Recursive Quick Sort
def quick_sort(data: list) -> list:
    """
    >>> for data in ([2, 1, 0], [2.2, 1.1, 0], "quick_sort"):
    ...     quick_sort(data) == sorted(data)
    True
    True
    True
    """
    if len(data) <= 1:
        return data
    else:
        return [
            *quick_sort([e for e in data[1:] if e <= data[0]]),
            data[0],
            *quick_sort([e for e in data[1:] if e > data[0]]),
        ]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Selection Sort
"""
This is a pure Python implementation of the selection sort algorithm

For doctests run following command:
python -m doctest -v selection_sort.py
or
python3 -m doctest -v selection_sort.py

For manual testing run:
python selection_sort.py
"""


def selection_sort(collection):
    """Pure implementation of the selection sort algorithm in Python
    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: the same collection ordered by ascending


    Examples:
    >>> selection_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> selection_sort([])
    []

    >>> selection_sort([-2, -5, -45])
    [-45, -5, -2]
    """

    length = len(collection)
    for i in range(length - 1):
        least = i
        for k in range(i + 1, length):
            if collection[k] < collection[least]:
                least = k
        if least != i:
            collection[least], collection[i] = (collection[i], collection[least])
    return collection


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(selection_sort(unsorted))

## Shell Sort
"""
https://en.wikipedia.org/wiki/Shellsort#Pseudocode
"""


def shell_sort(collection):
    """Pure implementation of shell sort algorithm in Python
    :param collection:  Some mutable ordered collection with heterogeneous
    comparable items inside
    :return:  the same collection ordered by ascending

    >>> shell_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]
    >>> shell_sort([])
    []
    >>> shell_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    # Marcin Ciura's gap sequence

    gaps = [701, 301, 132, 57, 23, 10, 4, 1]
    for gap in gaps:
        for i in range(gap, len(collection)):
            insert_value = collection[i]
            j = i
            while j >= gap and collection[j - gap] > insert_value:
                collection[j] = collection[j - gap]
                j -= gap
            if j != i:
                collection[j] = insert_value
    return collection


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(shell_sort(unsorted))

## Shrink Shell Sort
"""
This function implements the shell sort algorithm
which is slightly faster than its pure implementation.

This shell sort is implemented using a gap, which
shrinks by a certain factor each iteration. In this
implementation, the gap is initially set to the
length of the collection. The gap is then reduced by
a certain factor (1.3) each iteration.

For each iteration, the algorithm compares elements
that are a certain number of positions apart
(determined by the gap). If the element at the higher
position is greater than the element at the lower
position, the two elements are swapped. The process
is repeated until the gap is equal to 1.

The reason this is more efficient is that it reduces
the number of comparisons that need to be made. By
using a smaller gap, the list is sorted more quickly.
"""


def shell_sort(collection: list) -> list:
    """Implementation of shell sort algorithm in Python
    :param collection:  Some mutable ordered collection with heterogeneous
    comparable items inside
    :return:  the same collection ordered by ascending

    >>> shell_sort([3, 2, 1])
    [1, 2, 3]
    >>> shell_sort([])
    []
    >>> shell_sort([1])
    [1]
    """

    # Choose an initial gap value
    gap = len(collection)

    # Set the gap value to be decreased by a factor of 1.3
    # after each iteration
    shrink = 1.3

    # Continue sorting until the gap is 1
    while gap > 1:
        # Decrease the gap value
        gap = int(gap / shrink)

        # Sort the elements using insertion sort
        for i in range(gap, len(collection)):
            temp = collection[i]
            j = i
            while j >= gap and collection[j - gap] > temp:
                collection[j] = collection[j - gap]
                j -= gap
            collection[j] = temp

    return collection


if __name__ == "__main__":
    import doctest

    doctest.testmod()

## Slowsort
"""
Slowsort is a sorting algorithm. It is of humorous nature and not useful.
It's based on the principle of multiply and surrender,
a tongue-in-cheek joke of divide and conquer.
It was published in 1986 by Andrei Broder and Jorge Stolfi
in their paper Pessimal Algorithms and Simplexity Analysis
(a parody of optimal algorithms and complexity analysis).

Source: https://en.wikipedia.org/wiki/Slowsort
"""
#from __future__ import annotations


def slowsort(sequence: list, start: int | None = None, end: int | None = None) -> None:
    """
    Sorts sequence[start..end] (both inclusive) in-place.
    start defaults to 0 if not given.
    end defaults to len(sequence) - 1 if not given.
    It returns None.
    >>> seq = [1, 6, 2, 5, 3, 4, 4, 5]; slowsort(seq); seq
    [1, 2, 3, 4, 4, 5, 5, 6]
    >>> seq = []; slowsort(seq); seq
    []
    >>> seq = [2]; slowsort(seq); seq
    [2]
    >>> seq = [1, 2, 3, 4]; slowsort(seq); seq
    [1, 2, 3, 4]
    >>> seq = [4, 3, 2, 1]; slowsort(seq); seq
    [1, 2, 3, 4]
    >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]; slowsort(seq, 2, 7); seq
    [9, 8, 2, 3, 4, 5, 6, 7, 1, 0]
    >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]; slowsort(seq, end = 4); seq
    [5, 6, 7, 8, 9, 4, 3, 2, 1, 0]
    >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]; slowsort(seq, start = 5); seq
    [9, 8, 7, 6, 5, 0, 1, 2, 3, 4]
    """
    if start is None:
        start = 0

    if end is None:
        end = len(sequence) - 1

    if start >= end:
        return

    mid = (start + end) // 2

    slowsort(sequence, start, mid)
    slowsort(sequence, mid + 1, end)

    if sequence[end] < sequence[mid]:
        sequence[end], sequence[mid] = sequence[mid], sequence[end]

    slowsort(sequence, start, end - 1)


if __name__ == "__main__":
    from doctest import testmod

    testmod()

## Stooge Sort
def stooge_sort(arr):
    """
    Examples:
    >>> stooge_sort([18.1, 0, -7.1, -1, 2, 2])
    [-7.1, -1, 0, 2, 2, 18.1]

    >>> stooge_sort([])
    []
    """
    stooge(arr, 0, len(arr) - 1)
    return arr


def stooge(arr, i, h):
    if i >= h:
        return

    # If first element is smaller than the last then swap them
    if arr[i] > arr[h]:
        arr[i], arr[h] = arr[h], arr[i]

    # If there are more than 2 elements in the array
    if h - i + 1 > 2:
        t = (int)((h - i + 1) / 3)

        # Recursively sort first 2/3 elements
        stooge(arr, i, (h - t))

        # Recursively sort last 2/3 elements
        stooge(arr, i + t, (h))

        # Recursively sort first 2/3 elements
        stooge(arr, i, (h - t))


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(stooge_sort(unsorted))

## Strand Sort
import operator


def strand_sort(arr: list, reverse: bool = False, solution: list | None = None) -> list:
    """
    Strand sort implementation
    source: https://en.wikipedia.org/wiki/Strand_sort

    :param arr: Unordered input list
    :param reverse: Descent ordering flag
    :param solution: Ordered items container

    Examples:
    >>> strand_sort([4, 2, 5, 3, 0, 1])
    [0, 1, 2, 3, 4, 5]

    >>> strand_sort([4, 2, 5, 3, 0, 1], reverse=True)
    [5, 4, 3, 2, 1, 0]
    """
    _operator = operator.lt if reverse else operator.gt
    solution = solution or []

    if not arr:
        return solution

    sublist = [arr.pop(0)]
    for i, item in enumerate(arr):
        if _operator(item, sublist[-1]):
            sublist.append(item)
            arr.pop(i)

    #  merging sublist into solution list
    if not solution:
        solution.extend(sublist)
    else:
        while sublist:
            item = sublist.pop(0)
            for i, xx in enumerate(solution):
                if not _operator(item, xx):
                    solution.insert(i, item)
                    break
            else:
                solution.append(item)

    strand_sort(arr, reverse, solution)
    return solution


if __name__ == "__main__":
    assert strand_sort([4, 3, 5, 1, 2]) == [1, 2, 3, 4, 5]
    assert strand_sort([4, 3, 5, 1, 2], reverse=True) == [5, 4, 3, 2, 1]

## Tim Sort
def binary_search(lst, item, start, end):
    if start == end:
        return start if lst[start] > item else start + 1
    if start > end:
        return start

    mid = (start + end) // 2
    if lst[mid] < item:
        return binary_search(lst, item, mid + 1, end)
    elif lst[mid] > item:
        return binary_search(lst, item, start, mid - 1)
    else:
        return mid


def insertion_sort(lst):
    length = len(lst)

    for index in range(1, length):
        value = lst[index]
        pos = binary_search(lst, value, 0, index - 1)
        lst = lst[:pos] + [value] + lst[pos:index] + lst[index + 1 :]

    return lst


def merge(left, right):
    if not left:
        return right

    if not right:
        return left

    if left[0] < right[0]:
        return [left[0], *merge(left[1:], right)]

    return [right[0], *merge(left, right[1:])]


def tim_sort(lst):
    """
    >>> tim_sort("Python")
    ['P', 'h', 'n', 'o', 't', 'y']
    >>> tim_sort((1.1, 1, 0, -1, -1.1))
    [-1.1, -1, 0, 1, 1.1]
    >>> tim_sort(list(reversed(list(range(7)))))
    [0, 1, 2, 3, 4, 5, 6]
    >>> tim_sort([3, 2, 1]) == insertion_sort([3, 2, 1])
    True
    >>> tim_sort([3, 2, 1]) == sorted([3, 2, 1])
    True
    """
    length = len(lst)
    runs, sorted_runs = [], []
    new_run = [lst[0]]
    sorted_array = []
    i = 1
    while i < length:
        if lst[i] < lst[i - 1]:
            runs.append(new_run)
            new_run = [lst[i]]
        else:
            new_run.append(lst[i])
        i += 1
    runs.append(new_run)

    for run in runs:
        sorted_runs.append(insertion_sort(run))
    for run in sorted_runs:
        sorted_array = merge(sorted_array, run)

    return sorted_array


def main():
    lst = [5, 9, 10, 3, -4, 5, 178, 92, 46, -18, 0, 7]
    sorted_lst = tim_sort(lst)
    print(sorted_lst)


if __name__ == "__main__":
    main()

## Topological Sort
"""Topological Sort."""

#     a
#    / \
#   b  c
#  / \
# d  e
edges = {"a": ["c", "b"], "b": ["d", "e"], "c": [], "d": [], "e": []}
vertices = ["a", "b", "c", "d", "e"]


def topological_sort(start, visited, sort):
    """Perform topological sort on a directed acyclic graph."""
    current = start
    # add current to visited
    visited.append(current)
    neighbors = edges[current]
    for neighbor in neighbors:
        # if neighbor not in visited, visit
        if neighbor not in visited:
            sort = topological_sort(neighbor, visited, sort)
    # if all neighbors visited add current to sort
    sort.append(current)
    # if all vertices haven't been visited select a new one to visit
    if len(visited) != len(vertices):
        for vertice in vertices:
            if vertice not in visited:
                sort = topological_sort(vertice, visited, sort)
    # return sort
    return sort


if __name__ == "__main__":
    sort = topological_sort("a", [], [])
    print(sort)

## Tree Sort
"""
Tree_sort algorithm.

Build a BST and in order traverse.
"""


class Node:
    # BST data structure
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def insert(self, val):
        if self.val:
            if val < self.val:
                if self.left is None:
                    self.left = Node(val)
                else:
                    self.left.insert(val)
            elif val > self.val:
                if self.right is None:
                    self.right = Node(val)
                else:
                    self.right.insert(val)
        else:
            self.val = val


def inorder(root, res):
    # Recursive traversal
    if root:
        inorder(root.left, res)
        res.append(root.val)
        inorder(root.right, res)


def tree_sort(arr):
    # Build BST
    if len(arr) == 0:
        return arr
    root = Node(arr[0])
    for i in range(1, len(arr)):
        root.insert(arr[i])
    # Traverse BST in order.
    res = []
    inorder(root, res)
    return res


if __name__ == "__main__":
    print(tree_sort([10, 1, 3, 2, 9, 14, 13]))

## Unknown Sort
"""
Python implementation of a sort algorithm.
Best Case Scenario : O(n)
Worst Case Scenario : O(n^2) because native Python functions:min, max and remove are
already O(n)
"""


def merge_sort(collection):
    """Pure implementation of the fastest merge sort algorithm in Python

    :param collection: some mutable ordered collection with heterogeneous
    comparable items inside
    :return: a collection ordered by ascending

    Examples:
    >>> merge_sort([0, 5, 3, 2, 2])
    [0, 2, 2, 3, 5]

    >>> merge_sort([])
    []

    >>> merge_sort([-2, -5, -45])
    [-45, -5, -2]
    """
    start, end = [], []
    while len(collection) > 1:
        min_one, max_one = min(collection), max(collection)
        start.append(min_one)
        end.append(max_one)
        collection.remove(min_one)
        collection.remove(max_one)
    end.reverse()
    return start + collection + end


if __name__ == "__main__":
    user_input = input("Enter numbers separated by a comma:\n").strip()
    unsorted = [int(item) for item in user_input.split(",")]
    print(*merge_sort(unsorted), sep=",")

## Wiggle Sort
"""
Wiggle Sort.

Given an unsorted array nums, reorder it such
that nums[0] < nums[1] > nums[2] < nums[3]....
For example:
if input numbers = [3, 5, 2, 1, 6, 4]
one possible Wiggle Sorted answer is [3, 5, 1, 6, 2, 4].
"""


def wiggle_sort(nums: list) -> list:
    """
    Python implementation of wiggle.
    Example:
    >>> wiggle_sort([0, 5, 3, 2, 2])
    [0, 5, 2, 3, 2]
    >>> wiggle_sort([])
    []
    >>> wiggle_sort([-2, -5, -45])
    [-45, -2, -5]
    >>> wiggle_sort([-2.1, -5.68, -45.11])
    [-45.11, -2.1, -5.68]
    """
    for i, _ in enumerate(nums):
        if (i % 2 == 1) == (nums[i - 1] > nums[i]):
            nums[i - 1], nums[i] = nums[i], nums[i - 1]

    return nums


if __name__ == "__main__":
    print("Enter the array elements:")
    array = list(map(int, input().split()))
    print("The unsorted array is:")
    print(array)
    print("Array after Wiggle sort:")
    print(wiggle_sort(array))