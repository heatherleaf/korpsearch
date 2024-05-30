
from typing import BinaryIO
from collections.abc import Iterator, Sequence

from disk import DiskIntArray
from pathlib import Path

def find_msb(n: int) -> tuple[int, int]:
    assert n >= 0
    msb = -1
    mask = 0
    while (n & mask) != n:
        msb += 1
        mask = 2*mask+1

    return msb, (mask+1) >> 1

def split(n: int) -> tuple[int, int]:
    msb, mask = find_msb(n)

    return msb, n & (mask-1)

def bits(n: int, msb: int) -> Iterator[int]:
    while msb != -1:
        yield (n >> msb) & 1
        msb -= 1

def elias_gamma_code(n: int) -> Iterator[int]:
    msb, remainder = split(n)

    for _ in range(msb): yield 0
    yield 1
    yield from bits(remainder, msb-1)

def elias_delta_code(n: int) -> Iterator[int]:
    msb, remainder = split(n)
    yield from elias_gamma_code(msb+1)
    yield from bits(remainder, msb-1)

def write_bits(file: BinaryIO, bits: Iterator[int]) -> None:
    n = 0
    val = 0

    for bit in bits:
        val = (val << 1) | bit
        n += 1

        if n == 8:
            file.write(val.to_bytes(1, 'little'))
            n = 0
            val = 0

    if n != 0:
        val = val << (8 - n)
        file.write(val.to_bytes(1, 'little'))

def delta_code_array(arr: Sequence[int]) -> Iterator[int]:
    last = None

    for n in arr:
        if last is None or n <= last:
            # Start of new set - we can detect this via the .starts file
            # yield from bits(n, (arr._elemsize*8)-1)

            yield from elias_delta_code(1)
            yield from elias_delta_code(n+1)
        else:
            yield from elias_delta_code(n - last + 1)

        last = n

def write_delta_coded_array(file: BinaryIO, arr: Sequence[int]) -> None:
    write_bits(file, delta_code_array(arr))


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        print(name)
        with DiskIntArray(Path(name)) as arr:
            with open(name.replace(".ia", ".delta"), "wb") as file:
                write_delta_coded_array(file, arr)
