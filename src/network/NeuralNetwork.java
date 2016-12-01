package network;

import com.google.gson.Gson;

import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {

    private int numHiddenLayers;
    private int inputLayerSize;
    private double learningRate;

    private NeuronLayer[] layers;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numHiddenLayers, double learningRate) {
        this.inputLayerSize = inputSize;
        this.numHiddenLayers = numHiddenLayers;
        this.learningRate = learningRate;

        layers = new NeuronLayer[numHiddenLayers + 1];
        if (numHiddenLayers == 0) {
            layers[0] = new NeuronLayer(outputSize, inputSize);
        } else {
            layers[0] = new NeuronLayer(hiddenSize, inputSize);
            for (int i = 1; i < numHiddenLayers; i++) {
                layers[i] = new NeuronLayer(hiddenSize, hiddenSize);
            }
            layers[numHiddenLayers] = new NeuronLayer(outputSize, hiddenSize);
        }
    }

    /**
     * Fills this network with random values to provide a starting point.
     */
    public void fillWithRandomValues() {
        Random random = new Random();
        for (NeuronLayer layer : layers) {
            layer.fillWithRandomValues(random);
        }
    }

    /**
     * Moves forward through the network and calculates neuron output values.  Each layer is calculated from the outputs
     * of the previous layer.  Calling this will overwrite any existing output values stored within neurons and neuron
     * layers.
     * @param inputs the data set that is used as the first layer's inputs.
     * @return the final output set of the output layer.
     */
    public double[] propagateForward(double[] inputs) {
        if (inputs.length != inputLayerSize) {
            throw new IllegalArgumentException("Input array must be of size: " + inputLayerSize);
        }
        double[] currentOutputs = inputs;
        for (int i = 0; i < layers.length; i++) {
            layers[i].calculateOutputs(currentOutputs);
            currentOutputs = layers[i].getOutputs();
        }

        return Arrays.copyOf(currentOutputs, currentOutputs.length);
    }

    /**
     * Trains the network with the quadratic cost backpropagation algorithm.  For each array in the training data set,
     * the network calculates the output, calculates the error of each neuron by comparing the output to the label
     * array, calculates gradients for each neuron in the network via backpropagation, and adds the gradients to the
     * sums that are stored within each neuron.  Gradients are not applied to the weights or biases of the network until
     * a mini batch is completed or the training data set it exhausted.  When all training inputs have been used to
     * train the network, an epoch is complete.
     * @param trainingData the set of training data.
     * @param trainingLabels the set of labels associated with the training data.
     * @param numEpochs the number of training epochs.
     * @param miniBatchSize the mini batch size, after which gradients are applied.
     */
    public void trainFromInputSet(double[][] trainingData, double[][] trainingLabels, int numEpochs, int miniBatchSize) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // As we move through the sets, we calculate weight and bias gradients from the inputs of the previous layer
            // and the weights of the following layer.  We do not apply any of these gradients until we have calculated
            // them for the whole input set.
            for (int setNumber = 0; setNumber < trainingData.length; setNumber++) {
                double[] input = trainingData[setNumber];
                double[] expected = trainingLabels[setNumber];
                propagateForward(input);

                // Calculate the gradients of the output layer first.
                NeuronLayer outputLayer = layers[layers.length - 1];
                outputLayer.calculateErrorValues(expected);
                if (numHiddenLayers == 0) {
                    outputLayer.addToGradients(input);
                } else {
                    outputLayer.addToGradients(layers[layers.length - 2].getOutputs());
                }

                // Calculate the gradients of all hidden layers.
                for (int layerNum = layers.length - 2; layerNum >= 0; layerNum--) {
                    NeuronLayer currentLayer = layers[layerNum];
                    NeuronLayer nextLayer = layers[layerNum + 1];
                    double[] previousLayerOutputs = layerNum == 0 ? input : layers[layerNum - 1].getOutputs();
                    currentLayer.calculateErrorValues(nextLayer);
                    currentLayer.addToGradients(previousLayerOutputs);
                }

                // Applies the gradients that have been stored in each neuron.  This causes teh network to actually
                // learn from its mistakes. Calling this will clear the gradients from each neuron after they have been
                // applied to prevent adding the same gradients repeatedly.
                if (setNumber != 0 && setNumber % miniBatchSize == 0) {
                    for (NeuronLayer layer : layers) {
                        layer.applyGradients(learningRate);
                    }
                }
            }

            // Applies gradients one last time to ensure that the whole training set was used.  Calling this when the
            // neurons have no gradients has no effect.
            for (NeuronLayer layer : layers) {
                layer.applyGradients(learningRate);
            }
        }
    }

    /**
     * Returns a json representation of the network that can be saved to a file.
     * @param network the network to be saved.
     * @return a json representation of the network that can be saved to a file.
     */
    public static String saveToJson(NeuralNetwork network) {
        Gson gson = new Gson();
        return gson.toJson(network);
    }

    /**
     * Returns a neural network from a previously saved json.
     * @param json a json representation of a previously saved network.
     * @return a neural network from a previously saved json.
     */
    public static NeuralNetwork loadFromJson(String json) {
        return new Gson().fromJson(json, NeuralNetwork.class);
    }
}
