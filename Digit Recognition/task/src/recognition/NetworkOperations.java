package recognition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NetworkOperations {
    private MultilayerNetwork net;

    NetworkOperations(MultilayerNetwork net) {
        this.net = net;
    }

    public MultilayerNetwork learn(ArrayList<double[][]> trainingData, int epochs, int miniBatchSize, double eta) {
        for (int epo = 1; epo <= epochs; epo++) {
            Collections.shuffle(trainingData);

            for (int k = 0; k < trainingData.size() / miniBatchSize; k++) {
                update_mini_batch(trainingData.subList(k * miniBatchSize, (k + 1) * miniBatchSize), eta, miniBatchSize);
            }
        }

        return this.net;
    }

    private void update_mini_batch(List<double[][]> miniBatch, double eta, int miniBatchSize) {
        Nabla total = new Nabla(net);

        for (double[][] example : miniBatch) {
            total.add(backpropogate(example));
        }

        for (int layer = 0; layer < net.biases.length; layer++) {
            for (int bias = 0; bias < net.biases[layer].length; bias++) {
                net.biases[layer][bias] -= total.nablaB[layer][bias] * eta / miniBatchSize;
            }
        }

        for (int layer = 0; layer < net.weights.length; layer++) {
            for (int neuron = 0; neuron < net.weights[layer].length; neuron++) {
                for (int weight = 0; weight < net.weights[layer][neuron].length; weight++) {
                    net.weights[layer][neuron][weight] -= total.nablaW[layer][neuron][weight] * eta / miniBatchSize;
                }
            }
        }
    }

    private Nabla backpropogate(double[][] example) {
        double[][] nablaB = ArrayOperations.copy2DArray(net.biases);
        double[][][] nablaW = ArrayOperations.copy3DArray(net.weights);

        FeedForwardResult feed = feedForward(Arrays.copyOf(example, 28));

        double[] cdVec = costDerivativeVec(example[28][0], feed.activations[feed.activations.length - 1]);
        int last = nablaB.length - 1;

        for (int i = 0; i < nablaB[last].length; i++) {
            nablaB[last][i] = cdVec[i] * sigmoidPrime(feed.zs[feed.zs.length - 1][i]);
        }

        for (int j = 0; j < nablaW[last].length; j++) {
            for (int k = 0; k < nablaW[last][j].length; k++) {
                nablaW[last][j][k] = nablaB[last][j] * feed.activations[last][k];
            }
        }

        for (int i = 2; i <= nablaB.length; i++) {
            int layer = nablaB.length - i;
            double[] dot = dot(transpose(net.weights[layer + 1]), nablaB[layer + 1]);

            for (int j = 0; j < nablaB[layer].length; j++) {
                nablaB[layer][j] = sigmoidPrime(feed.zs[layer + 1][j]) * dot[j];
            }

            for (int j = 0; j < nablaW[layer].length; j++) {
                for (int k = 0; k < nablaW[layer][j].length; k++) {
                    nablaW[layer][j][k] = nablaB[layer][j] * feed.activations[layer][k];
                }
            }
        }

        Nabla result = new Nabla(net);

        result.nablaB = nablaB;
        result.nablaW = nablaW;

        return result;
    }

    private static double[][] transpose(double[][] array) {
        double[][] result = new double[array[0].length][];

        for (int inner = 0; inner < array[0].length; inner++) {
            result[inner] = new double[array.length];

            for (int outer = 0; outer < array.length; outer++) {
                result[inner][outer] = array[outer][inner];
            }
        }

        return result;
    }

    private static double[] costDerivativeVec(double classification, double[] activation) {
        double[] goal = new double[activation.length];
        Arrays.fill(goal, 0);
        goal[(int) classification] = 1;

        double[] dCost = new double[goal.length];

        for (int i = 0; i < goal.length; i++) {
            dCost[i] = activation[i] - goal[i];
        }

        return dCost;
    }

    void evaluate(ArrayList<double[][]> testData) {
        int numCorrect = 0;
        int total = testData.size();

        for (double[][] data : testData) {
            FeedForwardResult thisResult = feedForward(Arrays.copyOf(data, 28));

            if (thisResult.classification == data[28][0]) {
                numCorrect++;
            }
        }

        double accuracy = (double) numCorrect / total;

        System.out.printf("%d/%d, %.0f %s\n", numCorrect, total, accuracy * 100, "%");
    }

    FeedForwardResult feedForward(double[][] image) {
        double[][] activations = new double[net.numLayers][];
        double[][] zs = new double[net.numLayers][];
        double[] theseZs = new double[image.length * image[0].length];

        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[i].length; j++) {
                int k = i * image.length + j;
                theseZs[k] = image[i][j];
            }
        }

        zs[0] = theseZs;
        activations[0] = zs[0];
        double[] product;
        double sum;

        for (int i = 0; i < net.weights.length; i++) {
            product = dot(net.weights[i], activations[i]);
            zs[i + 1] = new double[product.length];
            activations[i + 1] = new double[product.length];

            for (int j = 0; j < product.length; j++) {
                sum = product[j] + net.biases[i][j];
                zs[i + 1][j] = sum;
                activations[i + 1][j] = sigmoid(sum);
            }
        }

        double max = 0;
        int classification = 0;

        for (int k = 0; k < activations[activations.length - 1].length; k++) {
            if (activations[activations.length - 1][k] > max) {
                max = activations[activations.length - 1][k];
                classification = k;
            }
        }

        FeedForwardResult result = new FeedForwardResult();

        result.classification = classification;
        result.zs = zs;
        result.activations = activations;

        return result;
    }

    private static double[] dot(double[][] matrixA, double[] matrixB) {
        double[] result = new double[matrixA.length];

        for (int outer = 0; outer < matrixA.length; outer++) {
            result[outer] = 0;

            for (int inner = 0; inner < matrixA[outer].length; inner++) {
                result[outer] += matrixA[outer][inner] * matrixB[inner];
            }
        }

        return result;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double sigmoidPrime(double x) {
        double s = sigmoid(x);

        return s * (1.0 - s);
    }
}
