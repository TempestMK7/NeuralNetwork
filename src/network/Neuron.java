package network;

import java.util.Random;

class Neuron {

    private int previousLayerSize;
    private double[] previousLayerWeights;
    private double bias;

    private transient double[] weightGradients;
    private transient double biasGradient;
    private transient int gradientSampleSize;
    private transient double outputValue;

    Neuron(int previousLayerSize) {
        this.previousLayerSize = previousLayerSize;
        this.previousLayerWeights = new double[previousLayerSize];
        weightGradients = new double[previousLayerSize];
    }

    /**
     * Fills the neuron with random values to provide a starting point for the network.
     * @param random the random number generator used by this network.
     */
    void fillWithRandomValues(Random random) {
        for (int i = 0; i < previousLayerSize; i++) {
            previousLayerWeights[i] = random.nextDouble() - 0.5;
        }
        bias = random.nextDouble() - 0.5;
    }

    /**
     * Returns the final output value of this neuron.
     * @return the final output value of this neuron.
     */
    double getOutput() {
        return outputValue;
    }

    /**
     * Returns the weight that this neuron places on the input of a specific neuron from the previous layer.
     * @param previousLayerIndex the index for the neuron in the previous layer.
     * @return the weight that this neuron places on the input of a specific neuron from the previous layer.
     */
    double getWeight(int previousLayerIndex) {
        return previousLayerWeights[previousLayerIndex];
    }

    /**
     * This calculates the output of the neuron as a function of its input values.  Each input is multiplied by the
     * weight that this neuron places on the input, then the summation is added to the neuron bias, then the output is
     * squashed and returned.
     * @param inputs the array of input values from the previous neuron layer.
     * @return the final output value of the neuron which will be a value from 0 to 1.
     */
    double calculateOutput(double[] inputs) {
        if (inputs.length != previousLayerWeights.length) {
            throw new IllegalArgumentException("Neuron can only accept inputs of size: " + previousLayerWeights.length);
        }

        double summation = 0.0;
        for (int i = 0; i < previousLayerWeights.length; i++) {
            summation += inputs[i] * previousLayerWeights[i];
        }

        outputValue = squash(summation, bias);
        return outputValue;
    }

    /**
     * Runs the summation of inputs and bias through the sigma function to "squash" the neuron output.  This will always
     * return a value from 0 to 1.
     * @param summation the summation of the input values multiplied by their weights.
     * @param bias the bias of the neuron.
     * @return the squashed value of the neuron output.
     */
    private static double squash(double summation, double bias) {
        return 1.0 / (1.0 + Math.exp((summation - bias) * -1.0));
    }

    /**
     * Clears the gradients that have been added to this neuron.  This should be called once gradients have been applied
     * to avoid applying the same gradients repeatedly.
     */
    private void clearGradients() {
        weightGradients = new double[previousLayerSize];
        biasGradient = 0;
        gradientSampleSize = 0;
    }

    /**
     * Takes an error value for this neuron and an array of inputs and adds the gradient values to the existing set of
     * gradient values.
     * @param error the error value for this neuron.
     * @param inputs the array of inputs for a specific data set.
     */
    void addToGradients(double error, double[] inputs) {
        if (weightGradients == null) weightGradients = new double[previousLayerSize];
        for (int i = 0; i < inputs.length; i++) {
            weightGradients[i] += error * inputs[i];
        }
        biasGradient += error;
        gradientSampleSize++;
    }

    /**
     * Finds the average gradients from all of the gradients that have been calculated thus far and applies them to the
     * weights and biases of this neuron.  This also clears the existing gradient totals when finished.
     * @param learningRate the learning rate for the network to be applied to the final gradient before it is applied to
     *                     the weights and bias.
     */
    void applyGradients(double learningRate) {
        for (int i = 0; i < previousLayerSize; i++) {
            previousLayerWeights[i] -= learningRate * weightGradients[i] / gradientSampleSize;
        }
        bias -= learningRate * biasGradient / gradientSampleSize;
        clearGradients();
    }
}
