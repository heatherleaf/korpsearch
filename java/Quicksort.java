
import java.util.Comparator;
import java.util.List;
import java.util.Arrays;

public class Quicksort<E> {
    Comparator<? super E> comparator;
    PivotSelector selector;
    int cutoff;

    Quicksort(Comparator<? super E> cmp, PivotSelector selector, int cutoff) {
        this.comparator = cmp;
        this.selector = selector;
        this.cutoff = cutoff;
    }

    public void sort(List<E> array) {
        sort(array, 0, array.size());
    }

    public void sort(List<E> array, int lo, int hi) {
        int size = hi - lo;
        if (size == 2) {
            if (comparator.compare(array.get(lo), array.get(lo+1)) > 0)
                swap(array, lo, lo+1);
        } else if (size <= cutoff) {
            builtinSort(array, lo, size);
        } else if (size > 2) {
            int mid = partition(array, lo, hi);
            sort(array, lo, mid);
            sort(array, mid+1, hi);
        }
    }

    private void builtinSort(List<E> array, int offset, int size) {
        @SuppressWarnings("unchecked")
        E[] sortingArray = (E[]) new Object[size];
        for (int i=0; i < size; i++)
            sortingArray[i] = array.get(i + offset);
        Arrays.sort(sortingArray, comparator);
        for (int i=0; i < size; i++)
            array.set(i + offset, sortingArray[i]);
    }

    private int partition(List<E> array, int lo, int hi) {
        int piv = selector.pivot(array, lo, hi, comparator);
        if (piv != lo) 
            swap(array, lo, piv);
        E pivot = array.get(lo);

        int i = lo + 1, j = hi - 1;
        while (i <= j) {
            while (i <= j && comparator.compare(array.get(i), pivot) < 0)
                i++;
            while (i <= j && comparator.compare(pivot, array.get(j)) < 0)
                j--;
            if (i <= j)
                swap(array, i++, j--);
        }

        swap(array, lo, j);
        return j;
    }        


    interface PivotSelector {
        public <E> int pivot(List<E> array, int lo, int hi, Comparator<? super E> cmp);
    }

    public static class TakeFirstPivotSelector implements PivotSelector {
        public <E> int pivot(List<E> array, int lo, int hi, Comparator<? super E> cmp) {
            return lo;
        }
    }

    public static class RandomPivotSelector implements PivotSelector {
        public <E> int pivot(List<E> array, int lo, int hi, Comparator<? super E> cmp) {
            int piv = (int) (lo + Math.random() * (hi - lo));
            return piv;
        }
    }

    public static class MedianOfThreePivotSelector implements PivotSelector {
        public <E> int pivot(List<E> array, int lo, int hi, Comparator<? super E> cmp) {
            int i = lo, j = hi-1, k = (i+j) / 2;
            E ei = array.get(i), ej = array.get(j), ek = array.get(k);
            if (cmp.compare(ei, ej) < 0) {        // ei < ej:
                if (cmp.compare(ej, ek) < 0)      //   ei < ej < ek
                    return j;
                else if (cmp.compare(ei, ek) < 0) //   ei < ek <= ej
                    return k;
                else                              //   ek < ei < ej
                    return i;
            } else {                              // ej <= ei:
                if (cmp.compare(ei, ek) < 0)      //   ej <= ei < ek
                    return i;
                else if (cmp.compare(ej, ek) < 0) //   ej < ek <= ei
                    return k;
                else                              //   ek <= ej <= ei
                    return j;
            }
        }
    }

    private static <E> void swap(List<E> array, int i, int j) {
        array.set(j, array.set(i, array.get(j)));
    }

}
