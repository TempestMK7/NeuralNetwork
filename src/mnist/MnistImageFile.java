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

/**
 * MNIST database image file. Contains additional header information for the number of rows and columns per each entry.
 */
public class MnistImageFile extends MnistDbFile {
    private int rows;
    private int cols;

    /**
     * Creates new MNIST database image file ready for reading.
     * @param name the system-dependent filename
     * @param mode the access mode
     * @throws IOException the IOException.
     */
    public MnistImageFile(String name, String mode) throws  IOException {
        super(name, mode);

        // read header information
        rows = readInt();
        cols = readInt();
    }

    public int[] readImageAsDoubleArray() throws IOException {
        int[] dat = new int[getRows() * getCols()];
        for (int x = 0; x < getRows(); x++) {
            for (int y = 0; y < getCols(); y++) {
                int index = (x * getCols()) + y;
                dat[index] = readUnsignedByte() > 30 ? 1 : -1;
            }
        }
        return dat;
    }

    /**
     * Read the specified number of images from the current position, to a byte[nImages][rows*cols]
     * Note that MNIST data set is stored as unsigned bytes; this method returns signed bytes without conversion
     * (i.e., same bits, but requires conversion before use)
     * @param nImages Number of images
     */
    public byte[][] readImagesUnsafe(int nImages) throws IOException{
        byte[][] out = new byte[nImages][0];
        for( int i=0; i<nImages; i++){
            out[i] = new byte[rows*cols];
            read(out[i]);
        }
        return out;
    }

    @Override
    protected int getMagicNumber() {
        return 2051;
    }

    /**
     * Number of rows per image.
     * @return int
     */
    public int getRows() {
        return rows;
    }

    /**
     * Number of columns per image.
     * @return int
     */
    public int getCols() {
        return cols;
    }

    @Override
    public int getEntryLength() {
        return cols * rows;
    }

    @Override
    public int getHeaderSize() {
        return super.getHeaderSize() + 8; // to more integers - rows and columns
    }
}
