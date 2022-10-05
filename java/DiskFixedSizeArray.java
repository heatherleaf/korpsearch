
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.RandomAccess;
import java.io.IOException;

import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;

public class DiskFixedSizeArray extends AbstractList<DiskFixedSizeArray.Record> implements AutoCloseable, RandomAccess {
    FileChannel channel;
    MappedByteBuffer buffer;
    int recordSize;
    int arraySize;

    DiskFixedSizeArray(String arrayPath, int recordSize) throws IOException {
        this(Path.of(arrayPath), recordSize);
    }

    DiskFixedSizeArray(Path arrayPath, int recordSize) throws IOException {
        if (recordSize <= 0)
            throw new IllegalArgumentException();
        channel = (FileChannel) Files.newByteChannel(arrayPath, StandardOpenOption.READ, StandardOpenOption.WRITE);
        buffer = channel.map(FileChannel.MapMode.READ_WRITE, 0, channel.size());
        this.recordSize = recordSize;
        this.arraySize = (int) (channel.size() / recordSize);
    }

    @Override
    public boolean isEmpty() {
        return arraySize == 0;
    }

    @Override
    public int size() {
        return arraySize;
    }

    @Override
    public Record get(int i) {
        byte[] record = new byte[recordSize];
        buffer.position(i * recordSize).get(record);
        return new Record(record);
    }

    @Override
    public Record set(int i, Record val) {
        Record old = get(i);
        buffer.position(i * recordSize).put(val.record);
        return old;
    }

    @Override
    public void close() throws IOException {
        buffer.force();
        channel.close();
    }

    public class Record implements Comparable<Record> {
        public byte[] record;
        Record(byte[] record) {
            this.record = record;
        }
        @Override
        public int compareTo(Record other) {
            int res = Arrays.compareUnsigned(this.record, other.record);
            return res;
        }
    }


    public static void main(String[] args) throws IOException {
        if (args.length < 2) reportError("Too few arguments");
        if (args.length > 4) reportError("Too many arguments");

        String arrayPath = args[0];
    
        int recordSize = 0;
        try {
            recordSize = Integer.parseInt(args[1]);
        } catch(NumberFormatException e) {
            reportError("Record size must be a positive integer");
        }

        Quicksort.PivotSelector selector = new Quicksort.RandomPivotSelector();
        if (args.length > 2) {
            char s = args[2].charAt(0);
            selector = (s == 'f' ? new Quicksort.TakeFirstPivotSelector() :
                        s == 'r' ? new Quicksort.RandomPivotSelector() :
                        s == 'm' ? new Quicksort.MedianOfThreePivotSelector() :
                        reportError("Unknown pivot selector"));
        }

        int cutoff = 1000;
        if (args.length > 3) {
            try {
                cutoff = Integer.parseInt(args[3]);
            } catch(NumberFormatException e) {
                reportError("Cutoff must be a positive integer");
            }
        }

        Quicksort<Record> sorter = new Quicksort<>(Comparator.naturalOrder(), selector, cutoff);

        try (
            DiskFixedSizeArray array = new DiskFixedSizeArray(arrayPath, recordSize);
        ) {
            sorter.sort(array);
        }
    }

    public static <E> E reportError(String err) {
        System.err.println(
            "Usage: java DiskFixedSizeArray path-to-diskarray record-size [pivot-selector [cutoff]]\n" +
            "Where: pivot-selector = take (f)irst / (r)andom / (m)edian-of-three\n" + 
            "       cutoff = cutoff for calling Java's built-in sort\n\n" +
            "Sorts the given file, which consists of fixed-size byte records.\n" + 
            "The records are stored in big-endian order (when viewed as integers).\n"
        );
        throw new IllegalArgumentException(err);
    }
}
