package recognition;

import java.util.Arrays;

class ArrayOperations {
    static void add2DDoubleArray(double[][] array1, double[][] array2) {
        for (int i = 0; i < array1.length; i++) {
            for (int j = 0; j < array1[i].length; j++) {
                array2[i][j] = array1[i][j] + array2[i][j];
            }
        }
    }

    static void add3DDoubleArray(double[][][] array1, double[][][] array2) {
        for (int i = 0; i < array1.length; i++) {
            for (int j = 0; j < array1[i].length; j++) {
                for (int k = 0; k < array1[i][j].length; k++) {
                    array2[i][j][k] += array1[i][j][k];
                }
            }
        }
    }

    static double[][][] copy3DArray(double[][][] arrayToCopy) {
        double[][][] copy = new double[arrayToCopy.length][][];

        for (int i = 0; i < arrayToCopy.length; i++) {
            copy[i] = new double[arrayToCopy[i].length][];

            for (int j = 0; j < arrayToCopy[i].length; j++) {
                copy[i][j] = Arrays.copyOf(arrayToCopy[i][j], arrayToCopy[i][j].length);
                Arrays.fill(copy[i][j], 0);
            }
        }

        return copy;
    }

    static double[][] copy2DArray(double[][] arrayToCopy) {
        double[][] copy = new double[arrayToCopy.length][];

        for (int i = 0; i < arrayToCopy.length; i++) {
            copy[i] = Arrays.copyOf(arrayToCopy[i], arrayToCopy[i].length);
            Arrays.fill(copy[i], 0);
        }

        return copy;
    }
}
