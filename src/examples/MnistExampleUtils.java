package examples;

import mnist.MnistFileUtils;
import network.Network;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Scanner;

/**
 * This file contains some methods that can be sued to quickly create, train, improve, and test a neural network against
 * the MNIST handwriting recognition database.
 */
public class MnistExampleUtils {

    private static final String MNIST_NETWORK_FILE_NAME = "mnist-network.txt";
    private static final int MNIST_INPUT_SIZE = 28*28;
    private static final int MNIST_OUTPUT_SIZE = 10;

    private static final NumberFormat FORMAT = new DecimalFormat("0.000");

    /**
     * Creates a new neural network and trains it against the MNIST handwriting recognition database.  The resulting
     * network is saved to a file which can be reopened for later use/testing.
     */
    public static void createNewMNISTNetwork(int[] hiddenLayers, int trainingEpochs, float learningRate, int numThreads,
                                             int calculationsPerThread) {
        // These values must have been previously saved by the MnistFileUtils methods.
        float[][] trainingInput = MnistFileUtils.readMNISTTrainingImages();
        float[][] trainingLabels = MnistFileUtils.readMNISTTrainingLabels();
        float[][] validationInput = MnistFileUtils.readMNISTTestImages();
        float[][] validationLabels = MnistFileUtils.readMNISTTestLabels();

        int[] layerSizes = new int[hiddenLayers.length + 2];
        System.arraycopy(hiddenLayers, 0, layerSizes, 1, hiddenLayers.length);
        layerSizes[0] = MNIST_INPUT_SIZE;
        layerSizes[layerSizes.length - 1] = MNIST_OUTPUT_SIZE;
        Network network = new Network(layerSizes);
        network.validate(validationInput, validationLabels, numThreads);
        for (int i = 0; i < trainingEpochs; i++) {
            network.learn(trainingInput, trainingLabels, learningRate, numThreads, calculationsPerThread);
            network.validate(validationInput, validationLabels, numThreads);
            try {
                FileWriter writer = new FileWriter(MNIST_NETWORK_FILE_NAME);
                writer.write(Network.saveToJson(network));
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void improveExistingMNISTNetwork(int trainingEpochs, float learningRate, int numThreads,
                                                   int calculationsPerThread) {
        // These values must have been previously saved by the MnistFileUtils methods.
        float[][] trainingInput = MnistFileUtils.readMNISTTrainingImages();
        float[][] trainingLabels = MnistFileUtils.readMNISTTrainingLabels();
        float[][] validationInput = MnistFileUtils.readMNISTTestImages();
        float[][] validationLabels = MnistFileUtils.readMNISTTestLabels();
        Network network = null;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(MNIST_NETWORK_FILE_NAME));
            String json = reader.readLine();
            network = Network.loadFromJson(json);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (network == null) return;
        network.validate(validationInput, validationLabels, numThreads);
        for (int i = 0; i < trainingEpochs; i++) {
            network.learn(trainingInput, trainingLabels, learningRate, numThreads, calculationsPerThread);
            network.validate(validationInput, validationLabels, numThreads);
            try {
                FileWriter writer = new FileWriter(MNIST_NETWORK_FILE_NAME);
                writer.write(Network.saveToJson(network));
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Opens the previously saved network and tests it against the MNIST handwriting recognition database.
     */
    public static void testExistingMNISTNetwork(int numThreads) {
        float[][] validationInput = MnistFileUtils.readMNISTTestImages();
        float[][] validationLabels = MnistFileUtils.readMNISTTestLabels();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(MNIST_NETWORK_FILE_NAME));
            String json = reader.readLine();
            Network network = Network.loadFromJson(json);
            network.validate(validationInput, validationLabels, numThreads);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void showcaseNetwork(boolean onlyFailures) {
        Network network;
        Scanner kb = new Scanner(System.in);
        try {
            BufferedReader reader = new BufferedReader(new FileReader(MNIST_NETWORK_FILE_NAME));
            String json = reader.readLine();
            network = Network.loadFromJson(json);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        float[][] validationInput = MnistFileUtils.readMNISTTestImages();
        float[][] validationLabels = MnistFileUtils.readMNISTTestLabels();
        if (validationInput == null || validationLabels == null) return;

        for (int image = 0; image < validationInput.length; image++) {
            int label = evaluateLabel(validationLabels[image], false);
            int guess = evaluateLabel(network.ask(validationInput[image]), false);
            if (onlyFailures && label == guess) continue;
            drawImage(validationInput[image]);
            kb.nextLine();
            System.out.println("Label: " + label);
            System.out.println("Guess: " + guess);
            evaluateLabel(network.ask(validationInput[image]), true);
            kb.nextLine();
        }
    }

    public static void drawImage(float[] image) {
        for (int i = 0; i < 784; i++) {
            if (i % 28 == 0 && i != 0) System.out.print("\n");
            System.out.print(image[i] == 1 ? "▓" : "░");
        }
    }

    public static int evaluateLabel(float[] label, boolean readArray) {
        if (readArray) System.out.print("Raw Output: ");
        int biggest = 0;
        for (int i = 0; i < label.length; i++) {
            if (readArray) System.out.print(FORMAT.format(label[i]) + ", ");
            if (label[i] > label[biggest]) biggest = i;
        }
        if (readArray) System.out.print("\n");
        return biggest;
    }
}
