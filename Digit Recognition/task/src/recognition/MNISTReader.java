package recognition;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.stream.Stream;

class MNISTReader {
    private static File[] getFiles(String directory) {
        File dir = new File(directory);

        return dir.listFiles();
    }

    static ArrayList<double[][]> getAllNumbers() {
        File[] files = getFiles("data");
        ArrayList<double[][]> result = new ArrayList<double[][]>();

        for (File file : files) {
            double[][] image = getGridFromFile(file);
            result.add(image);
        }

        return result;
    }

    private static double[][] getGridFromFile(File file) {
        double[][] image = new double[29][28];

        try {
            FileReader fileReader = new FileReader(file);
            Scanner fileScanner = new Scanner(fileReader);
			for (int i = 0; i < 28; i++) {

				double[] line = Stream.of(fileScanner.nextLine().split("\\s+"))
						.mapToDouble(Double::parseDouble)
						.toArray();

				for (int j = 0; j < 28; j++) {
					image[i][j] = line[j] / 256.0;
				}
			}

            if (fileScanner.hasNextInt()) {
                double[] labelArray = new double[1];
                labelArray[0] = (double) fileScanner.nextInt();
                image[28] = labelArray;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return image;
    }

    static double[][] getImageFromFile(String file) {
        double[][] image = new double[28][28];

        try {
            FileReader fileReader = new FileReader(file);
            Scanner fileScanner = new Scanner(fileReader);

            for (int i = 0; i < 28; i++) {

                double[] line = Stream.of(fileScanner.nextLine().split("\\s+"))
                        .mapToDouble(Double::parseDouble)
                        .toArray();

                for (int j = 0; j < 28; j++) {
                    image[i][j] = line[j] / 256.0;
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return image;
    }
}