package recognition;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

class MultilayerNetwork implements Serializable {

    int numLayers = 0;
    int[] sizes;
    double[][] biases;
    double[][][] weights;
    private static final long serialVersionUID = 123456321L;

    MultilayerNetwork(int[] sizes) {
        this.sizes = sizes;
        numLayers = sizes.length;
        randomInitialization();
    }

    void randomInitialization() {
        int[] internal = Arrays.copyOfRange(sizes, 1, sizes.length);

        biases = new double[numLayers - 1][];
        weights = new double[numLayers - 1][][];

        int biasIndex = 0;

        Random rand = new Random();

        for (int size : internal) {
            biases[biasIndex] = new double[size];

            for (int i = 0; i < size; i++) {
                biases[biasIndex][i] = rand.nextGaussian();
            }

            biasIndex++;
        }

        for (int i = 0; i < numLayers - 1; i++) {
            weights[i] = new double[sizes[i + 1]][];

            for (int j = 0; j < sizes[i + 1]; j++) {

                weights[i][j] = new double[sizes[i]];
                for (int k = 0; k < sizes[i]; k++) {
                    weights[i][j][k] = rand.nextGaussian();
                }
            }
        }
    }
}
