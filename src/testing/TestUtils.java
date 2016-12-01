package testing;

import network.NeuralNetwork;

public class TestUtils {
    /**
     * Returns the number of inputs on which the network guessed correctly.
     * @param network the network.
     * @param input the input data set, should be the validation input set.
     * @param labels the label set for the provided inputs.
     * @return the number of inputs on which the network guessed correctly.
     */
    public static int testNetworkCorrectness(NeuralNetwork network, double[][] input, double[][] labels) {
        int correct = 0;
        for (int i = 0; i < input.length; i++) {
            double[] actual = network.propagateForward(input[i]);
            double[] expect = labels[i];
            if (isCorrect(actual, expect)) correct++;
        }
        return correct;
    }

    /**
     * Returns whether or not the network guessed correctly on an input array.
     * @param networkOutput the input array.
     * @param label the label array.
     * @return whether or not the network guessed correctly.
     */
    private static boolean isCorrect(double[] networkOutput, double[] label) {
        int index = 0;
        for (int i = 0; i < networkOutput.length; i++) {
            if (networkOutput[i] > networkOutput[index]) index = i;
        }
        return label[index] == 1;
    }

    /**
     * Returns the total offset of the network from "perfect" across an input set.
     * @param network the network.
     * @param input the input data set, should be the validation input set.
     * @param labels the label set for the input set.
     * @return the total offset of the network from "perfect."
     */
    public static double testNetworkAccuracy(NeuralNetwork network, double[][] input, double[][] labels) {
        double totalError = 0;
        for (int i = 0; i < input.length; i++) {
            double[] actual = network.propagateForward(input[i]);
            double[] expected = labels[i];
            totalError += differenceArrays(actual, expected);
        }
        return totalError;
    }

    /**
     * Returns the difference between two arrays.
     * @param networkOutput the output of a network.
     * @param label the label array.
     * @return the difference between two arrays.
     */
    private static double differenceArrays(double[] networkOutput, double[] label) {
        double total = 0;
        for (int i = 0; i < networkOutput.length; i++) {
            total += Math.pow(networkOutput[i] - label[i], 2);
        }
        return Math.sqrt(total);
    }

    /**
     * Prints all of the network's statistics to the console.
     * @param network the network.
     * @param input the input set, should be the validation set.
     * @param labels the labels for the input set.
     */
    public static void printStatistics(NeuralNetwork network, double[][] input, double[][] labels) {
        int totalCount = input.length;
        int numCorrect = testNetworkCorrectness(network, input, labels);
        double accuracy = testNetworkAccuracy(network, input, labels) / 10000;
        System.out.println("Network chose correctly in " + numCorrect + " / " + totalCount
                + " with an average error of " + accuracy + " per input.");
    }
}
