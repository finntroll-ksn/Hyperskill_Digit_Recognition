package recognition;

class Nabla {
    double[][] nablaB;
    double[][][] nablaW;

    Nabla(MultilayerNetwork net) {
        nablaB = ArrayOperations.copy2DArray(net.biases);
        nablaW = ArrayOperations.copy3DArray(net.weights);
    }

    void add(Nabla nToAdd) {
        ArrayOperations.add2DDoubleArray(nToAdd.nablaB, nablaB);
        ArrayOperations.add3DDoubleArray(nToAdd.nablaW, nablaW);
    }
}
