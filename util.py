
import time

def log(output, verbose, start=None):
    if verbose:
        if start:
            duration = time.time()-start
            print(output.ljust(100), f"{duration//60:4.0f}:{duration%60:05.2f}")
        else:
            print(output)


def bytesify(s):
    return s.encode() if isinstance(s, str) else bytes(s)
