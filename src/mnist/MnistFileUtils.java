package mnist;

import com.google.gson.Gson;

import java.io.*;

public class MnistFileUtils {

    private static final String TRAINING_IMAGES = "training-images.txt";
    private static final String TRAINING_LABELS = "training-labels.txt";
    private static final String TEST_IMAGES = "test-images.txt";
    private static final String TEST_LABELS = "test-labels.txt";

    /**
     * Writes the MNIST handwriting recognition training files to a more quickly readable series of text files.
     */
    private static void writeMNISTTrainingFiles() {
        System.out.println("Reading MNIST training files.");
        try {
            MnistManager manager = new MnistManager("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
            int numPixels = manager.getImages().getCols() * manager.getImages().getRows();
            int[][] inputImages = new int[MnistManager.NUM_EXAMPLES][numPixels];
            int[][] inputExpected = new int[MnistManager.NUM_EXAMPLES][10];
            for (int i = 0; i < MnistManager.NUM_EXAMPLES; i++) {
                manager.setCurrent(i);
                int[] image = manager.readImageAsDoubleArray();
                int[] expected = new int[10];
                int labelValue = manager.readLabel();
                expected[labelValue] = 1;

                inputImages[i] = image;
                inputExpected[i] = expected;
            }

            System.out.println("Writing MNIST training files.");
            Gson gson = new Gson();
            FileWriter writer = new FileWriter(new File(TRAINING_IMAGES));
            writer.write(gson.toJson(inputImages));
            writer.close();

            writer = new FileWriter(new File(TRAINING_LABELS));
            writer.write(gson.toJson(inputExpected));
            writer.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Writes the MNIST handwriting recognition validation files to a more quickly readable series of text files.
     */
    private static void writeMNISTTestFiles() {
        System.out.println("Reading MNIST validation files.");
        try {
            MnistManager manager = new MnistManager("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", false);
            int numPixels = manager.getImages().getCols() * manager.getImages().getRows();
            int[][] testImages = new int[MnistManager.NUM_EXAMPLES_TEST][numPixels];
            int[][] testExpected = new int[MnistManager.NUM_EXAMPLES_TEST][10];

            for (int i = 0; i < MnistManager.NUM_EXAMPLES_TEST; i++) {
                manager.setCurrent(i);
                int[] image = manager.readImageAsDoubleArray();
                int[] expected = new int[10];
                int labelValue = manager.readLabel();
                expected[labelValue] = 1;

                testImages[i] = image;
                testExpected[i] = expected;
            }

            System.out.println("Writing MNIST validation files.");
            Gson gson = new Gson();
            FileWriter writer = new FileWriter(new File(TEST_IMAGES));
            writer.write(gson.toJson(testImages));
            writer.close();

            writer = new FileWriter(new File(TEST_LABELS));
            writer.write(gson.toJson(testExpected));
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Checks to see if the MNIST files exist and generates them if they do not.
     */
    private static void ensureFilesPresent() {
        File trainingImages = new File(TRAINING_IMAGES);
        File trainingLabels = new File(TRAINING_LABELS);
        File testImages = new File(TEST_IMAGES);
        File testLabels = new File(TEST_LABELS);

        if (!trainingImages.exists() || !trainingLabels.exists()) writeMNISTTrainingFiles();
        if (!testImages.exists() || !testLabels.exists()) writeMNISTTestFiles();
    }

    /**
     * Returns the training inputs from the MNIST handwriting recognition database.
     * @return the training inputs from the MNIST handwriting recognition database.
     */
    public static double[][] readMNISTTrainingImages() {
        ensureFilesPresent();
        System.out.println("Opening MNIST training inputs.");
        Gson gson = new Gson();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(TRAINING_IMAGES));
            String json = reader.readLine();
            return gson.fromJson(json, double[][].class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Returns the training labels from the MNIST handwriting recognition database.
     * @return the training labels from the MNIST handwriting recognition database.
     */
    public static double[][] readMNISTTrainingLabels() {
        ensureFilesPresent();
        System.out.println("Opening MNIST training labels.");
        Gson gson = new Gson();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(TRAINING_LABELS));
            String json = reader.readLine();
            return gson.fromJson(json, double[][].class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Returns the validation inputs from the MNIST handwriting recognition database.
     * @return the validation inputs from the MNIST handwriting recognition database.
     */
    public static double[][] readMNISTTestImages() {
        ensureFilesPresent();
        System.out.println("Opening MNIST validation inputs.");
        Gson gson = new Gson();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(TEST_IMAGES));
            String json = reader.readLine();
            return gson.fromJson(json, double[][].class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Returns the validation labels from the MNIST handwriting recognition database.
     * @return the validation labels from the MNIST handwriting recognition database.
     */
    public static double[][] readMNISTTestLabels() {
        ensureFilesPresent();
        System.out.println("Opening MNIST validation labels.");
        Gson gson = new Gson();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(TEST_LABELS));
            String json = reader.readLine();
            return gson.fromJson(json, double[][].class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
