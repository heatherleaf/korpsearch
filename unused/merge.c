#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "merge.h"

static inline int intersection_core(const char *in1, int len1, const char *in2, int len2, char *out, int size) {
    int i = 0, j = 0, k = 0;
    for (; i < len1 && j < len2;) {
        int x = 0, y = 0;
        memcpy(&x, &in1[i], size);
        memcpy(&y, &in2[j], size);

        if (x < y) i += size;
        else if (x > y) j += size;
        else {
            memcpy(&out[k], &x, size);
            i += size;
            j += size;
            k += size;
        }
    }

    return k;
}

int intersection(const char *in1, int len1, const char *in2, int len2, char *out, int size) {
    switch(size) {
        case 1: return intersection_core(in1, len1, in2, len2, out, size);
        case 2: return intersection_core(in1, len1, in2, len2, out, size);
        case 3: return intersection_core(in1, len1, in2, len2, out, size);
        case 4: return intersection_core(in1, len1, in2, len2, out, size);
        case 5: return intersection_core(in1, len1, in2, len2, out, size);
        case 6: return intersection_core(in1, len1, in2, len2, out, size);
        case 7: return intersection_core(in1, len1, in2, len2, out, size);
        case 8: return intersection_core(in1, len1, in2, len2, out, size);
        default: return 0;
    }
}

#if 0
#define LEN1 1319731
#define LEN2 1866242

int arr1[LEN1];
int arr2[LEN2];
int arr3[LEN1 > LEN2 ? LEN1 : LEN2];

int main() {
    srand(12345678);

    clock_t before, after;

    before = clock();
    int base = 0;
    for (int i = 0; i < LEN1; i++) {
        base = base + 1 + rand() % 8;
        arr1[i] = base;
    }

    base = 0;
    for (int i = 0; i < LEN2; i++) {
        base = base + 1 + rand() % 6;
        arr2[i] = base;
    }
    after = clock();
    printf("Initialisation took %.2fs\n", (float)(after - before) / CLOCKS_PER_SEC);

    before = clock();
    int s = sizeof(int);
    int k = intersection((const char *)arr1, LEN1*s, (const char *)arr2, LEN2*s, (char *)arr3, s);
    after = clock();
    printf("%d items in common\n", k/s);
    printf("Intersection took %.2fs\n", (float)(after - before) / CLOCKS_PER_SEC);
}
#endif
