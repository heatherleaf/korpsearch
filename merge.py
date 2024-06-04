
from array import array
from typing import Union

from disk import DiskIntArray, DiskIntArrayBuilder


IndexSetValuesType = Union[DiskIntArray, 'array[int]']
IndexSetValuesBuilder = Union[DiskIntArray, DiskIntArrayBuilder, 'array[int]']


def merge(
            arr1: IndexSetValuesType, start1: int, length1: int, offset1: int, 
            arr2: IndexSetValuesType, start2: int, length2: int, offset2: int, 
            result: IndexSetValuesBuilder, take_first: bool, take_second: bool, take_common: bool,
        ) -> None:

    i = start1
    j = start2
    end1 = start1 + length1
    end2 = start2 + length2
    x = arr1[i] - offset1
    y = arr2[j] - offset2

    while True:
        if x < y: 
            if take_first:
                result.append(x)
            i += 1
            if i >= end1:
                break
            x = arr1[i] - offset1

        elif y < x: 
            if take_second:
                result.append(y)
            j += 1
            if j >= end2:
                break
            y = arr2[j] - offset2

        else:
            if take_common:
                result.append(x)
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
            result.append(arr1[i] - offset1)
            i += 1
    
    if take_second:
        while j < end2:
            result.append(arr2[j] - offset2)
            j += 1


