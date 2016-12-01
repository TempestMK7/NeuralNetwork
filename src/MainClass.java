import examples.BinaryClassificationExampleUtils;
import examples.MnistExampleUtils;

public class MainClass {

    public static void main(String[] args) {
        //BinaryClassificationExampleUtils.testLearningWithBinary(8, 3, 2.0, 100000, 10);
        //MnistExampleUtils.createNewMNISTNetwork(60, 2, 0.5, 5, 10);
        MnistExampleUtils.improveExistingMNISTNetwork(5, 5, 10);
    }
}
