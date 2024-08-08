

def merge(
        arr1: memoryview, start1: int, length1: int, offset1: int, 
        arr2: memoryview, start2: int, length2: int, offset2: int, 
        result: memoryview, take_first: bool, take_second: bool, take_common: bool,
    ) -> int:
    """
    Merge two sorted arrays A (arr1) and B (arr2).
    The result array must have enough space for all elements.
    Returns the size of the merged result (so the result array can be truncated).

    * If take_first  is True then elements in A - B are included.
    * If take_second is True then elements in B - A are included.
    * If take_common is True then elements in A & B are included.

    You can get the following set operations (among others):

    Operation              take_first     take_second    take_common
    union        (A | B)   True           True           True
    intersection (A & B)   False          False          True
    difference   (A - B)   True           False          False
    """

    i = start1
    j = start2
    k = 0
    end1 = start1 + length1
    end2 = start2 + length2
    x = arr1[i] - offset1
    y = arr2[j] - offset2

    while True:
        if x < y: 
            if take_first:
                result[k] = x
                k += 1
            i += 1
            if i >= end1: 
                break
            x = arr1[i] - offset1

        elif y < x: 
            if take_second:
                result[k] = y
                k += 1
            j += 1
            if j >= end2: 
                break
            y = arr2[j] - offset2

        else:
            if take_common:
                result[k] = x
                k += 1
            i += 1
            if i >= end1: 
                break
            j += 1
            if j >= end2: 
                break
            x = arr1[i] - offset1
            y = arr2[j] - offset2

    if take_first:
        while i < end1:
            result[k] = arr1[i] - offset1
            k += 1
            i += 1
    
    if take_second:
        while j < end2:
            result[k] = arr2[j] - offset2
            k += 1
            j += 1

    return k


def sort(arr: memoryview, start: int, length: int, result: memoryview) -> None:
    sorted_arr: list[int] = sorted(arr[start : start + length])
    for i in range(length):
        result[i] = sorted_arr[i]

