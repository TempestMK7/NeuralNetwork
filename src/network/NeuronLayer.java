package network;

import java.util.Random;

class NeuronLayer {
    private int layerSize;
    private Neuron[] neurons;

    private transient double[] neuronOutputs;
    private transient double[] neuronErrors;

    NeuronLayer(int layerSize, int previousLayerSize) {
        this.layerSize = layerSize;
        neurons = new Neuron[layerSize];
        neuronOutputs = new double[layerSize];
        neuronErrors = new double[layerSize];

        for (int i = 0; i < layerSize; i++) {
            neurons[i] = new Neuron(previousLayerSize);
        }
    }

    /**
     * Returns the number of neurons in this layer.
     * @return the number of neurons in this layer.
     */
    private double getSize() {
        return layerSize;
    }

    /**
     * Returns an array of the outputs of each neuron in this layer.
     * @return an array of the outputs of each neuron in this layer.
     */
    double[] getOutputs() {
        return neuronOutputs;
    }

    /**
     * Returns an array of the error values stored by each neuron in this layer.
     * @return an array of the error values stored by each neuron in this layer.
     */
    private double[] getErrors() {
        return neuronErrors;
    }

    /**
     * Gets the weight that a neuron in this layer applies to a specific neuron in the previous layer.  This is
     * necessary for the backpropagation algorithm to compute error gradients of hidden layers.
     * @param previousLayerIndex the index of the neuron from the previous layer.
     * @param thisLayerIndex the index of the neuron in this layer.
     * @return the weight that the neuron in this layer applies to the neuron in the previous layer.
     */
    private double getWeight(int previousLayerIndex, int thisLayerIndex) {
        return neurons[thisLayerIndex].getWeight(previousLayerIndex);
    }

    /**
     * Fills the neurons in this layer with random weights and biases.
     * @param random the random number generator.
     */
    void fillWithRandomValues(Random random) {
        for (int i = 0; i < layerSize; i++) {
            neurons[i].fillWithRandomValues(random);
        }
    }

    /**
     * Moves through the layer and calculates the output for each neuron.
     * @param inputs the inputs from which the neurons calculate their outputs.
     */
    void calculateOutputs(double[] inputs) {
        if (neuronOutputs == null) {
            neuronOutputs = new double[layerSize];
        }
        for (int i = 0; i < layerSize; i++) {
            neuronOutputs[i] = neurons[i].calculateOutput(inputs);
        }
    }

    /**
     * Moves through the layer and calculates the error value for each neuron.  This should only be called if the layer
     * is the output layer or last layer in the network.
     * @param expectedOutputs the label array that the network's answer is compared against.
     */
    void calculateErrorValues(double[] expectedOutputs) {
        if (neuronErrors == null) neuronErrors = new double[layerSize];
        for (int i = 0; i < layerSize; i++) {
            double actual = neurons[i].getOutput();
            double error = actual - expectedOutputs[i];
            neuronErrors[i] = error * actual * (1.0 - actual);
        }
    }

    /**
     * Moves through the layer and calculates the error value for each neuron.  This should only be called for hidden
     * layers and not for the output layer.
     * @param nextLayer the next neuron layer from which error values are gathered.  This is necessary for
     *                  backpropagation computations.
     */
    void calculateErrorValues(NeuronLayer nextLayer) {
        if (neuronErrors == null) neuronErrors = new double[layerSize];
        for (int myIndex = 0; myIndex < layerSize; myIndex++) {
            double actual = neurons[myIndex].getOutput();
            double error = 0;
            double[] nextLayerErrors = nextLayer.getErrors();
            for (int nextLayerIndex = 0; nextLayerIndex < nextLayer.getSize(); nextLayerIndex++) {
                error += nextLayerErrors[nextLayerIndex] * nextLayer.getWeight(myIndex, nextLayerIndex);
            }
            neuronErrors[myIndex] = error * actual * (1.0 - actual);
        }
    }

    /**
     * Calculates the error gradients for the neurons in this layer from the outputs of the previous neuron layer.
     * Note that this method does not apply the gradients.
     * @param previousLayerOutputs the outputs of the previous neuron layer.
     */
    void addToGradients(double[] previousLayerOutputs) {
        for (int i = 0; i < layerSize; i++) {
            neurons[i].addToGradients(neuronErrors[i], previousLayerOutputs);
        }
    }

    /**
     * Applies the previously calculated gradents to all of the weights and biases of neurons in this layer.  This
     * resets the gradients for each neuron to prevent them from being added repeatedly.
     * @param learningRate the learning rate of the network.
     */
    void applyGradients(double learningRate) {
        for (int i = 0; i < layerSize; i++) {
            neurons[i].applyGradients(learningRate);
        }
    }
}
