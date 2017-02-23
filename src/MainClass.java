import examples.MnistExampleUtils;

public class MainClass {

    public static void main(String[] args) {
        MnistExampleUtils.createNewMNISTNetwork(new int[]{2000, 1500, 1000, 500, 100}, 5, 1f, 5, 10);
        //MnistExampleUtils.improveExistingMNISTNetwork(18, 0.5f, 5, 10);
        //MnistExampleUtils.showcaseNetwork(false);
    }
}
