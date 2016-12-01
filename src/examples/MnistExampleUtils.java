package examples;

import mnist.MnistFileUtils;
import network.NeuralNetwork;
import testing.TestUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This file contains some methods that can be sued to quickly create, train, improve, and test a neural network against
 * the MNIST handwriting recognition database.
 */
public class MnistExampleUtils {

    private static final String MNIST_NETWORK_FILE_NAME = "mnist-network.txt";
    private static final int MNIST_INPUT_SIZE = 28*28;
    private static final int MNIST_OUTPUT_SIZE = 10;

    /**
     * Creates a new neural network and trains it against the MNIST handwriting recognition database.  The resulting
     * network is saved to a file which can be reopened for later use/testing.
     * @param hiddenLayerSize the number of neurons in each hidden layer.
     * @param numHiddenLayers the number of hidden layers.
     * @param learningRate the learning rate of the network.
     * @param numEpochs the number of training epochs.
     * @param miniBatchSize the mini batch size of each epoch.
     */
    public static void createNewMNISTNetwork(int hiddenLayerSize, int numHiddenLayers, double learningRate,
                                             int numEpochs, int miniBatchSize) {
        // These values must have been previously saved by the MnistFileUtils methods.
        double[][] trainingInput = MnistFileUtils.readMNISTTrainingImages();
        double[][] trainingLabels = MnistFileUtils.readMNISTTrainingLabels();
        double[][] validationInput = MnistFileUtils.readMNISTTestImages();
        double[][] validationLabels = MnistFileUtils.readMNISTTestLabels();

        NeuralNetwork network = new NeuralNetwork(MNIST_INPUT_SIZE, hiddenLayerSize, MNIST_OUTPUT_SIZE,
                numHiddenLayers, learningRate);
        network.fillWithRandomValues();

        TestUtils.printStatistics(network, validationInput, validationLabels);
        network.trainFromInputSet(trainingInput, trainingLabels, numEpochs, miniBatchSize);
        TestUtils.printStatistics(network, validationInput, validationLabels);
        try {
            FileWriter writer = new FileWriter(MNIST_NETWORK_FILE_NAME);
            writer.write(NeuralNetwork.saveToJson(network));
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void improveExistingMNISTNetwork(int saveInterval, int numEpochsPerInterval, int miniBatchSize) {
        // These values must have been previously saved by the MnistFileUtils methods.
        double[][] trainingInput = MnistFileUtils.readMNISTTrainingImages();
        double[][] trainingLabels = MnistFileUtils.readMNISTTrainingLabels();
        double[][] validationInput = MnistFileUtils.readMNISTTestImages();
        double[][] validationLabels = MnistFileUtils.readMNISTTestLabels();
        NeuralNetwork network = null;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(MNIST_NETWORK_FILE_NAME));
            String json = reader.readLine();
            network = NeuralNetwork.loadFromJson(json);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (network == null) return;

        TestUtils.printStatistics(network, validationInput, validationLabels);

        for (int i = 0; i < saveInterval; i++) {
            network.trainFromInputSet(trainingInput, trainingLabels, numEpochsPerInterval, miniBatchSize);
            TestUtils.printStatistics(network, validationInput, validationLabels);
            try {
                FileWriter writer = new FileWriter(MNIST_NETWORK_FILE_NAME);
                writer.write(NeuralNetwork.saveToJson(network));
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Opens the previously saved network and tests it against the MNIST handwriting recognition database.
     */
    public static void testExistingMNISTNetwork() {
        double[][] validationInput = MnistFileUtils.readMNISTTestImages();
        double[][] validationLabels = MnistFileUtils.readMNISTTestLabels();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(MNIST_NETWORK_FILE_NAME));
            String json = reader.readLine();
            NeuralNetwork network = NeuralNetwork.loadFromJson(json);
            TestUtils.printStatistics(network, validationInput, validationLabels);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
