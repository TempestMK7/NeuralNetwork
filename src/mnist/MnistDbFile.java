/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package mnist;

import java.io.IOException;
import java.io.RandomAccessFile;

public abstract class MnistDbFile extends RandomAccessFile {

    private int count;

    /**
     * Creates new instance and reads the header information.
     * @param name the system-dependent filename
     * @param mode the access mode
     * @throws IOException the IOException
     * @see RandomAccessFile
     */
    public MnistDbFile(String name, String mode) throws IOException {
        super(name, mode);
        if (getMagicNumber() != readInt()) {
            throw new RuntimeException("This MNIST DB file " + name + " should start with the number " + getMagicNumber() + ".");
        }
        count = readInt();
    }

    /**
     * MNIST DB files start with unique integer number.
     *
     * @return integer number that should be found in the beginning of the file.
     */
    protected abstract int getMagicNumber();

    /**
     * The current entry index.
     *
     * @return long
     * @throws IOException
     */
    public long getCurrentIndex() throws IOException {
        return (getFilePointer() - getHeaderSize()) / getEntryLength() + 1;
    }

    /**
     * Set the required current entry index.
     * @param curr the entry index
     */
    public void setCurrentIndex(long curr) {
        try {
            if (curr < 0 || curr > count) {
                throw new RuntimeException(curr + " is not in the range 0 to " + count);
            }
            seek(getHeaderSize() + curr * getEntryLength());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public int getHeaderSize() {
        return 8; // two integers
    }

    /**
     * Number of bytes for each entry.
     * Defaults to 1.
     * @return int
     */
    public int getEntryLength() {
        return 1;
    }

    /**
     * Move to the next entry.
     * @throws IOException
     */
    public void next() throws IOException {
        if (getCurrentIndex() < count) {
            skipBytes(getEntryLength());
        }
    }

    /**
     * Move to the previous entry.
     * @throws IOException
     */
    public void prev() throws IOException {
        if (getCurrentIndex() > 0) {
            seek(getFilePointer() - getEntryLength());
        }
    }

    public int getCount() {
        return count;
    }
}