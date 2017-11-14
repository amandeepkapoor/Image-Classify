package skymind.round2.convolution;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
//import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.model.VGG16;



import javax.imageio.ImageTranscoder;
import javax.xml.crypto.Data;
import java.io.*;
import java.util.Random;

/**
 * Created by tom hanlon on 11/7/16.
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=GLC8CIoHDnI
 * <p>
 * This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 * <p>
 * Instructions
 * Downloads a directory containing a testing and a training folder
 * each folder has 10 directories 0-9
 * in each directory are 28 * 28 grayscale pngs of handwritten digits
 * The training and testing directories will have directories 0-9 with
 * 28 * 28 PNG images of handwritten images
 * <p>
 * The code here shows how to use a ParentPathLabelGenerator to label the images as
 * they are read into the RecordReader
 * <p>
 * The pixel values are scaled to values between 0 and 1 using
 * ImagePreProcessingScaler
 * <p>
 * In this example a loop steps through 3 images and prints the DataSet to
 * the terminal. The expected output is the 28* 28 matrix of scaled pixel values
 * the list with the label for that image
 * and a list of the label values
 * <p>
 * This example also applies a Listener to the RecordReader that logs the path of each image read
 * You would not want to do this in production
 * The reason it is done here is to show that a handwritten image 3 (for example)
 * was read from directory 3,
 * has a matrix with the shown values
 * Has a label value corresponding to 3
 */

public class zooLearning {

    /** Data URL for downloading */
    //public static final String DATA_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz";

    /**
     * Location to save and extract the training/testing data
     */
    //public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "caltech/");

    public static final String File_Sub_Path = "src\\main\\resources\\101_ObjectCategories";


    private static Logger log = LoggerFactory.getLogger(zooLearning.class);

    public static void main(String[] args) throws Exception {
        /*
        image information
        28 * 28 grayscale
        grayscale implies single channel
        */
        int height = 224;
        int width = 224;
        int channels = 3;
        int seed = 123;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 50;
        int outputNum = 101;
        int epoch = 15;
        //NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(16384);
        //NativeOpsHolder.getInstance().getDeviceNativeOps().setTADThreshold(64);
        //System.out.println(System.getProperty("user.home"));

        /*
        This class downloadData() downloads the data
        stores the data in java's tmpdir
        15MB download compressed
        It will take 158MB of space when uncompressed
        The data can be downloaded manually here
        http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
         */


        //downloadData();

        // Define the File Paths
        //File trainData = new File(DATA_PATH + "/mnist_png/training");
        //File testData = new File(DATA_PATH + "/mnist_png/testing");
        File trainData = new File(System.getProperty("user.dir"),File_Sub_Path);

        // Define the FileSplit(PATH, ALLOWED FORMATS,random)


        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        //FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, labelMaker, 0, outputNum, batchSize);
        InputSplit[] filesInDirSplit = train.sample(pathFilter, 80, 20);
        InputSplit trainData1 = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        /*ImageTransform transform1 = new FlipImageTransform(randNumGen);
        ImageTransform transform2 = new FlipImageTransform(new Random(rngseed));
        ImageTransform warptranform = new WarpImageTransform(randNumGen,42);
        List<ImageTransform> tranforms = Arrays.asList(new ImageTransform[] {transform1, warptranform, transform2});*/


        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name

        recordReader.initialize(trainData1);

        // The LogRecordListener will log the path of each image read
        // used here for information purposes,
        // If the whole dataset was ingested this would place 60,000
        // lines in our logs
        // It will show up in the output with this format
        // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png

        //recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator

        //DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1

        //DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
        DataNormalization scalar = new VGG16ImagePreProcessor();
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        //MultipleEpochsIterator trainIter;
        scalar.fit(dataIter);
        dataIter.setPreProcessor(scalar);

        // In production you would loop through all the data
        // in this example the loop is just through 3
        // images for demonstration purposes
        /*for (int i = 1; i < 3; i++) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());

        }*/
        /*MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .seed(seed)
                .iterations(1)
                .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(.006)
                .updater(Updater.NESTEROVS)//new AdaDelta())
                .regularization(true).l2(1e-4)
                .convolutionMode(ConvolutionMode.Same).list()
                // block 1
                .layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}).name("image_array").stride(new int[]{1, 1})
                        .nIn(3)
                        .nOut(16).build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}).stride(new int[]{5, 5}).nIn(16).nOut(16)
                        .build())
                .layer(3, new BatchNormalization.Builder().build())
                .layer(4, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                        new int[] {2, 2}).build())
                .layer(6, new DropoutLayer.Builder(0.5).build())
                .layer(7, new DenseLayer.Builder().name("ffn2").nOut(256).build())
                .layer(7, new DenseLayer.Builder().name("ffn3").nOut(256).build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output").nOut(outputNum).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(height, width, channels))
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");

        //recordReader.initialize(trainData1);
        for (int j = 0; j < epoch; j++) {


            model.fit(dataIter);
        }
        recordReader.reset();*/
        /*for (ImageTransform transform: ) {

            System.out.println("Training on"+transform.getClass().toString());
            recordReader.initialize(trainData1,transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);

            //trainIter = new MultipleEpochsIterator(epoch, dataIter);
            for (int j = 0; j < epoch; j++) {
                model.fit(dataIter);
            }
        }*/

        ZooModel zooModel = new VGG16();
        //int[] shape = new int[]{3,300,200};
        //zooModel.setInputShape (shape);
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrainedNet.summary());
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(5e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .seed(seed)
            .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor("fc2")
            .removeVertexKeepConnections("predictions")
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(outputNum)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX).build(), "fc2")
            .build();

        //log.info("*****TRAIN MODEL********");
        //vgg16Transfer.setListeners(new ScoreIterationListener(10));

        //for(int i = 0; i<2; i++){

        //dataIter.setPreProcessor(new VGG16ImagePreProcessor());

        //}


        log.info("*****Test MODEL********");
        recordReader.reset();

        // The model trained on the training dataset split
        // now that it has trained we evaluate against the
        // test data of images the network has not seen

        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scalar.fit(testIter);
        testIter.setPreProcessor(scalar);

        // Create Eval object with 10 possible classes
        log.info("Before----------------");
        Evaluation eval = new Evaluation(outputNum);
        eval = vgg16Transfer.evaluate(testIter);
        log.info(eval.stats() + "\n");

        // Evaluate the network
        int iter = 0;
        while(dataIter.hasNext()){
            vgg16Transfer.fit(dataIter.next());
            //DataSet next = testIter.next();

            if (iter % 10 == 0){
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            //INDArray output = model.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            //eval.eval(next.getLabels(),output);
            iter++;
        }



    }


    /*
    Everything below here has nothing to do with your RecordReader,
    or DataVec, or your Neural Network
    The classes downloadData, getMnistPNG(),
    and extractTarGz are for downloading and extracting the data
     */
/*
    private static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "mnist_png";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (15MB)...");
            getMnistPNG();
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    public static void getMnistPNG() throws IOException {
        String tmpDirStr = System.getProperty("java.io.tmpdir");
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";

        if (tmpDirStr == null) {
            throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir");
        }
        String url = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
        File f = new File(archizePath);
        File dir = new File(tmpDirStr);
        if (!f.exists()) {
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            try (CloseableHttpResponse response = client.execute(new HttpGet(url))) {
                HttpEntity entity = response.getEntity();
                if (entity != null) {
                    try (FileOutputStream outstream = new FileOutputStream(f)) {
                        entity.writeTo(outstream);
                        outstream.flush();
                        outstream.close();
                    }
                }

            }
            System.out.println("Data downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing directory at " + f.getAbsolutePath());
        }

    }
*/
}







