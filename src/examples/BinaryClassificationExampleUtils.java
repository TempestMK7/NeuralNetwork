package examples;

import network.NeuralNetwork;
import testing.TestUtils;

/**
 * These methods can be used to build an train a NeuralNetwork against a very simple binary logic problem.  These
 * methods are only to be used in preliminary testing when the network needs to remain small rather than proof of
 * concept for the network process as a whole.
 */
public class BinaryClassificationExampleUtils {
    private static final int BINARY_INPUT_SIZE = 10;
    private static final int BINARY_OUTPUT_SIZE = 5;
    private static final int BINARY_DATA_SET_SIZE = 1000;

    /**
     * Creates a new network and tests it against a simple binary logic problem.
     */
    public static void testLearningWithBinary(int hiddenLayerSize, int numHiddenLayers, double learningRate,
                                              int numEpochs, int miniBatchSize) {
        double[][] inputs = generateTrainingInputs();
        double[][] labels = generateTrainingLabels();

        NeuralNetwork network = new NeuralNetwork(BINARY_INPUT_SIZE, hiddenLayerSize, BINARY_OUTPUT_SIZE,
                numHiddenLayers, learningRate);
        network.fillWithRandomValues();
        // Validating against the training set should be avoided when demonstrating that a network is effective.  This
        // network is only used for preliminary testing purposes.
        TestUtils.printStatistics(network, inputs, labels);
        network.trainFromInputSet(inputs, labels, numEpochs, miniBatchSize);
        TestUtils.printStatistics(network, inputs, labels);
    }

    /**
     * Returns the input set used in the binary problem.
     * @return the input set used in the binary problem.
     */
    private static double[][] generateTrainingInputs() {
        double[][] dataSet = new double[BINARY_DATA_SET_SIZE][BINARY_INPUT_SIZE];
        for (int i = 0; i < BINARY_DATA_SET_SIZE; i++) {
            int value = i;
            for (int j = 0; j < BINARY_INPUT_SIZE; j++) {
                dataSet[i][j] = value % 2;
                value /= 2;
            }
        }
        return dataSet;
    }

    /**
     * Returns the label set used in the binary problem.
     * @return the label set used in the binary problem.
     */
    private static double[][] generateTrainingLabels() {
        double[][] labels = new double[BINARY_DATA_SET_SIZE][BINARY_OUTPUT_SIZE];
        for (int i = 0; i < BINARY_DATA_SET_SIZE; i++) {
            int index = i / 200;
            labels[i][index] = 1;
        }
        return labels;
    }
}
