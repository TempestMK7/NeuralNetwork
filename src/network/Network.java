package network;

import com.google.gson.Gson;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;
import java.util.concurrent.CountDownLatch;

public class Network {

    private static final SimpleDateFormat FORMAT = new SimpleDateFormat("hh:mm:ss");

    private int[] layerSizes;
    private float[][][] weights;
    private float[][] biases;
    private int completedCycles;

    public Network(int[] layerSizes) {
        int numLayers = layerSizes.length - 1;
        if (numLayers < 1) throw new IllegalArgumentException("Network needs at least two layers to function.");

        this.layerSizes = layerSizes;
        weights = new float[numLayers][][];
        biases = new float[numLayers][];
        for (int i = 1; i < layerSizes.length; i++) {
            weights[i - 1] = new float[layerSizes[i]][layerSizes[i - 1]];
            biases[i - 1] = new float[layerSizes[i]];
        }

        Random random = new Random();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = random.nextFloat() - 0.5f;
                }
                biases[i][j] = random.nextFloat() - 0.5f;
            }
        }
    }

    /**
     * Runs the summation of inputs and bias through the sigma function to "squash" the neuron output.  This will always
     * return a value from 0 to 1.
     * @param summation the summation of the input values multiplied by their weights.
     * @param bias the bias of the neuron.
     * @return the squashed value of the neuron output.
     */
    private static float squash(float summation, float bias) {
        return 1.0f / (1.0f + (float) Math.exp((summation - bias) * -1.0));
    }

    /**
     * Asks the network for output based on the input array.
     * @param inputs the input array.
     * @return the network's classification of the input array.
     */
    public float[] ask(float[] inputs) {
        if (inputs.length != layerSizes[0]) {
            throw new IllegalArgumentException("Network can only accept inputs of size: " + layerSizes[0]);
        }

        float[] previousLayerOutputs = inputs;
        for (int i = 0; i < weights.length; i++) {
            float[] layerOuputs = new float[weights[i].length];
            for (int j = 0; j < layerOuputs.length; j++) {
                float summation = 0;
                for (int k = 0; k < weights[i][j].length; k++) {
                    summation += weights[i][j][k] * previousLayerOutputs[k];
                }
                layerOuputs[j] = squash(summation, biases[i][j]);
            }
            previousLayerOutputs = layerOuputs;
        }
        return previousLayerOutputs;
    }

    /**
     * Trains the network from the provided data set.  This will adjust all of the weights and biases within the network
     * using a standard back-propagation algorithm.  Weight and bias gradients are computed across several sets and
     * averaged together.  These averages are computed on separate threads, then added together on the main thread
     * before they are applied to the network.
     * @param trainingData the set of training data.
     * @param trainingLabels the labels for the training data.
     * @param learningRate the rate at which the network adjusts its weights and biases.  Each gradient is multiplied
     *                     by this number before it is applied.
     * @param numThreads the number of threads which are used to compute gradients.
     * @param rangePerThread the amount of data each thread will use to compute gradients.
     */
    public void learn(float[][] trainingData, float[][] trainingLabels, float learningRate,
                      int numThreads, int rangePerThread) {
        int rangeStart = 0;

        while (rangeStart < trainingData.length - 1) {
            // Start the runnables in separate threads and await.
            TrainingRunnable[] runnables = new TrainingRunnable[numThreads];
            CountDownLatch latch = new CountDownLatch(numThreads);
            for (int i = 0; i < numThreads; i++) {
                if (rangeStart == trainingData.length - 1) {
                    latch.countDown();
                    continue;
                }
                int rangeEnd = rangeStart + rangePerThread > trainingData.length ? trainingData.length - 1 : rangeStart + rangePerThread;
                runnables[i] = new TrainingRunnable(latch, trainingData, trainingLabels, rangeStart, rangeEnd);
                new Thread(runnables[i]).start();
                rangeStart = rangeEnd;
            }
            try {
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
                return;
            }

            // Reap the results.
            float[][][] weightGradients = null;
            float[][] biasGradients = null;
            for (TrainingRunnable runnable : runnables) {
                if (runnable == null) continue;
                float[][][] runnableGradients = runnable.weightGradients;
                float[][] runnableBiasGradients = runnable.biasGradients;
                if (weightGradients == null || biasGradients == null) {
                    weightGradients = runnableGradients;
                    biasGradients = runnableBiasGradients;
                } else {
                    for (int i = 0; i < weightGradients.length; i++) {
                        for (int j = 0; j < weightGradients[i].length; j++) {
                            for (int k = 0; k < weightGradients[i][j].length; k++) {
                                weightGradients[i][j][k] += runnableGradients[i][j][k];
                            }
                            biasGradients[i][j] += runnableBiasGradients[i][j];
                        }
                    }
                }
            }
            if (weightGradients == null || biasGradients == null) {
                throw new IllegalStateException("Gradient matrices were null.");
            }
            int updateInterval = numThreads * rangePerThread;

            // Apply results to weights and biases.
            for (int i = 0; i < layerSizes.length - 1; i++) {
                for (int j = 0; j < layerSizes[i + 1]; j++) {
                    for (int k = 0; k < layerSizes[i]; k++) {
                        weights[i][j][k] -= learningRate * weightGradients[i][j][k] / updateInterval;
                    }
                    weightGradients[i][j] = new float[layerSizes[i]];
                    biases[i][j] -= learningRate * biasGradients[i][j] / updateInterval;
                }
                biasGradients[i] = new float[layerSizes[i + 1]];
            }
        }

        completedCycles++;
    }

    /**
     * Validates the network against the provided data set.  Each row in the data set is run through the network and
     * the results are compared against the associated labels.  A message is printed to the console showing how many
     * rows were guessed correctly as well as the average error across the entire dataset.
     * @param testData the test data set.
     * @param testLabels the test data labels.
     * @param numThreads the number of threads to use during validation.
     */
    public void validate(float[][] testData, float[][] testLabels, int numThreads) {
        System.out.println(getTimeStamp() + ": Starting validation (" + completedCycles + " training cycles completed).");
        int numCorrect = 0;
        double error = 0;

        ValidationRunnable[] runnables = new ValidationRunnable[numThreads];
        CountDownLatch latch = new CountDownLatch(numThreads);
        int rangeStart = 0;
        for (int i = 0; i < numThreads; i++) {
            int rangeEnd = i == numThreads - 1 ? testData.length : (testData.length / numThreads) + rangeStart;
            runnables[i] = new ValidationRunnable(latch, testData, testLabels, rangeStart, rangeEnd);
            new Thread(runnables[i]).start();
            rangeStart = rangeEnd;
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
            return;
        }

        for (ValidationRunnable runnable : runnables) {
            numCorrect += runnable.numCorrect;
            error += runnable.totalError;
        }
        error /= testData.length;

        int successRate = (numCorrect * 100) / testData.length;
        System.out.println(getTimeStamp() + ": Network chose correctly in " + numCorrect + " / " + testData.length
                + " cases (" + successRate + "%) with an average error of " + error + " per input.");
    }

    /**
     * A convenience method that prints the current time stamp.
     * @return a String representation of the current time stamp.
     */
    private static String getTimeStamp() {
        return FORMAT.format(new Date());
    }

    /**
     * Returns whether or not the network guessed correctly on an input array.
     * @param networkOutput the input array.
     * @param label the label array.
     * @return whether or not the network guessed correctly.
     */
    private static boolean isCorrect(float[] networkOutput, float[] label) {
        int index = 0;
        for (int i = 0; i < networkOutput.length; i++) {
            if (networkOutput[i] > networkOutput[index]) index = i;
        }
        return label[index] > 0.5f;
    }

    /**
     * Returns the difference between two arrays.
     * @param networkOutput the output of a network.
     * @param label the label array.
     * @return the difference between two arrays.
     */
    private static double differenceArrays(float[] networkOutput, float[] label) {
        double total = 0;
        for (int i = 0; i < networkOutput.length; i++) {
            total += Math.pow(networkOutput[i] - label[i], 2);
        }
        return Math.sqrt(total);
    }

    /**
     * Returns a String containing a json representation of the network object.
     * @param network the network to be serialized.
     * @return a String containing a json representation of the network object.
     */
    public static String saveToJson(Network network) {
        return new Gson().toJson(network);
    }

    /**
     * Returns a network which has been constructed from a json.
     * @param json the json representaiton of the network.
     * @return a network which has been constructed from a json.
     */
    public static Network loadFromJson(String json) {
        return new Gson().fromJson(json, Network.class);
    }

    private final class TrainingRunnable implements Runnable {
        private CountDownLatch latch;

        private float[][][] weightGradients;
        private float[][] biasGradients;

        private float[][] trainingData;
        private float[][] trainingLabels;
        private int rangeStart;
        private int rangeEnd;

        private TrainingRunnable(CountDownLatch latch, float[][] trainingData, float[][] trainingLabels,
                                int rangeStart, int rangeEnd) {
            this.latch = latch;
            this.trainingData = trainingData;
            this.trainingLabels = trainingLabels;
            this.rangeStart = rangeStart;
            this.rangeEnd = rangeEnd;
        }

        @Override
        public void run() {
            int numLayers = weights.length;
            float[][] outputs = new float[numLayers][];
            float[][] errors = new float[numLayers][];
            float[] previousLayerOutputs;
            weightGradients = new float[numLayers][][];
            biasGradients = new float[numLayers][];

            // Initialize gradient matrices.
            for (int x = 0; x < numLayers; x++) {
                weightGradients[x] = new float[layerSizes[x + 1]][layerSizes[x]];
                biasGradients[x] = new float[layerSizes[x + 1]];
            }

            for (int setNumber = rangeStart; setNumber < rangeEnd; setNumber++) {
                float[] input = trainingData[setNumber];
                float[] label = trainingLabels[setNumber];
                // Feed-forward algorithm.
                previousLayerOutputs = input;
                for (int i = 0; i < numLayers; i++) {
                    int currentLayerSize = layerSizes[i + 1];
                    outputs[i] = new float[currentLayerSize];
                    for (int j = 0; j < currentLayerSize; j++) {
                        float summation = 0;
                        for (int k = 0; k < weights[i][j].length; k++) {
                            summation += weights[i][j][k] * previousLayerOutputs[k];
                        }
                        outputs[i][j] = squash(summation, biases[i][j]);
                    }
                    previousLayerOutputs = outputs[i];
                }

                // Calculate errors for the output layer.
                errors[errors.length - 1] = new float[layerSizes[layerSizes.length - 1]];
                for (int i = 0; i < errors[errors.length - 1].length; i++) {
                    float actual = outputs[outputs.length - 1][i];
                    float error = actual - label[i];
                    errors[errors.length - 1][i] = error * actual * (1.0f - actual);
                }

                // Calculate errors for the hidden layers.
                for (int i = numLayers - 2; i >= 0; i--) {
                    errors[i] = new float[layerSizes[i + 1]];
                    for (int j = 0; j < errors[i].length; j++) {
                        float error = 0;
                        for (int k = 0; k < errors[i + 1].length; k++) {
                            error += errors[i + 1][k] * weights[i + 1][k][j];
                        }
                        errors[i][j] = error * outputs[i][j] * (1.0f - outputs[i][j]);
                    }
                }

                // Calculate weight and bias gradients for each layer.
                previousLayerOutputs = input;
                for (int i = 0; i < numLayers; i++) {
                    for (int j = 0; j < layerSizes[i + 1]; j++) {
                        for (int k = 0; k < previousLayerOutputs.length; k++) {
                            weightGradients[i][j][k] += errors[i][j] * previousLayerOutputs[k];
                        }
                        biasGradients[i][j] += errors[i][j];
                    }
                    previousLayerOutputs = outputs[i];
                }
            }

            latch.countDown();
        }
    }

    private final class ValidationRunnable implements Runnable {
        private CountDownLatch latch;

        private int numCorrect;
        private double totalError;

        private float[][] validationData;
        private float[][] validationLabels;
        private int rangeStart;
        private int rangeEnd;

        private ValidationRunnable(CountDownLatch latch, float[][] validationData, float[][] validationLabels,
                                  int rangeStart, int rangeEnd) {
            this.latch = latch;
            this.validationData = validationData;
            this.validationLabels = validationLabels;
            this.rangeStart = rangeStart;
            this.rangeEnd = rangeEnd;
        }

        @Override
        public void run() {
            for (int i = rangeStart; i < rangeEnd; i++) {
                float[] output = ask(validationData[i]);
                if (isCorrect(output, validationLabels[i])) numCorrect++;
                totalError += differenceArrays(output, validationLabels[i]);
            }
            latch.countDown();
        }
    }
}
