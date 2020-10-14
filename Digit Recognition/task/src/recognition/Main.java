package recognition;

import java.io.*;
import java.util.*;

public class Main {
    private static Scanner scanner = new Scanner(System.in);
    private static MultilayerNetwork net;
    private static final String fileName = "net.txt";

    private static void printMenu() {
        System.out.printf(
                "%s\n%s\n%s\n",
                "1. Learn the network",
                "2. Guess all the numbers",
                "3. Guess number from text file",
                "Your choice:"
        );
    }

    private static void saveNet(MultilayerNetwork net) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(net);
        oos.close();
    }

    private static Object loadNet() throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(fileName);
        BufferedInputStream bis = new BufferedInputStream(fis);
        ObjectInputStream ois = new ObjectInputStream(bis);
        Object net = ois.readObject();
        ois.close();

        return net;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        printMenu();

        int choose = Integer.parseInt(scanner.nextLine());

        switch (choose) {
            case 1:
                System.out.println("Your choice: 1");
                System.out.println("Enter the sizes of the layers:");

                String[] inputSizes = scanner.nextLine().split("\\s+");

                System.out.println("Learning...");

                net = new MultilayerNetwork(
                        new int[]{
                                Integer.parseInt(inputSizes[0]),
                                Integer.parseInt(inputSizes[1]),
                                Integer.parseInt(inputSizes[2]),
                                Integer.parseInt(inputSizes[3])
                        });
                net = new NetworkOperations(net).learn(MNISTReader.getAllNumbers(), 30, 10, 3.0);
                saveNet(net);

                System.out.println("Done! Saved to the file.");
                break;
            case 2:
                System.out.println("Your choice: 2");
                System.out.println("Guessing...");
                System.out.println("The network predict accuracy:");

                net = (MultilayerNetwork) loadNet();
                new NetworkOperations(net).evaluate(MNISTReader.getAllNumbers());
                break;
            case 3:
                System.out.println("Your choice: 3");
                System.out.println("Enter filename:");

                net = (MultilayerNetwork) loadNet();

                FeedForwardResult result = new NetworkOperations(net).feedForward(MNISTReader.getImageFromFile(scanner.nextLine()));

                System.out.println("This number is " + result.classification);
                break;
            default:
                throw new IllegalArgumentException("Wrong command");
        }
    }
}
