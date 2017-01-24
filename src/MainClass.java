import examples.MnistExampleUtils;

public class MainClass {

    public static void main(String[] args) {
        MnistExampleUtils.createNewMNISTNetwork(new int[]{1000, 800, 600, 400, 200}, 10, 1f, 5, 10);
        MnistExampleUtils.improveExistingMNISTNetwork(10, 0.5f, 5, 10);
        MnistExampleUtils.improveExistingMNISTNetwork(10, 0.1f, 5, 10);
        MnistExampleUtils.improveExistingMNISTNetwork(10, 0.05f, 5, 10);
    }
}
