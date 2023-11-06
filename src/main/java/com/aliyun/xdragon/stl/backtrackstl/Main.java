package com.aliyun.xdragon.stl.backtrackstl;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import com.aliyun.xdragon.stl.backtrackstl.common.CircularQueue;
import com.aliyun.xdragon.stl.backtrackstl.common.CsvUtils;
import com.aliyun.xdragon.stl.backtrackstl.common.Data;
import com.aliyun.xdragon.stl.backtrackstl.common.DataPoint;
import com.aliyun.xdragon.stl.backtrackstl.common.Utils;

/**
 * @author Haoyu Wang
 * @date 2023/10/08
 */
public class Main {
    public static double trendMAE, seasonMAE, residualMAE, latencyUs;

    public static void main(String[] args) throws Exception {
        createEnv();
        Data.generateData();
        accuracyExp();
        latencyExp();
    }

    private static void createEnv() {
        new File("dataset").mkdir();
        new File("result").mkdir();
    }

    public static void accuracyExp() throws Exception {
        BackTrackSTL model = new BackTrackSTL(2, 200, 5, 4, 6, true);
        testBacktrackSTL(model, "dataset/synthetic.csv", "result/accuracy.csv");
        System.out.println("Accuracy Exp:");
        System.out.println("Trend MAE = " + trendMAE);
        System.out.println("Seasonality MAE = " + seasonMAE);
        System.out.println("Residual MAE = " + residualMAE);
        System.out.println();
    }

    public static void latencyExp() throws Exception {
        int[] period = {200, 400, 800, 1600, 3200, 6400, 12800};
        int times = 50;
        System.out.println("Latency Exp:");
        for (int i = 0; i < period.length; i++) {
            BackTrackSTL model = new BackTrackSTL(2, period[i], 5, 4, 6, true);
            //Cold start
            testBacktrackSTL(model, "dataset/scalability.csv", null);
            double latency = 0;
            for (int k = 0; k < times; k++) {
                testBacktrackSTL(model, "dataset/scalability.csv", null);
                latency += latencyUs;
            }
            System.out.println(period[i] + "," + (latency / times));
        }
    }

    public static void testBacktrackSTL(BackTrackSTL model, String input, String output) throws Exception {
        List<DataPoint> inputData = CsvUtils.fromCsv(input);
        double[] values = inputData.stream().mapToDouble(DataPoint::getValue).toArray();
        List<DataPoint> result = new ArrayList<>();

        int initLength = model.getInitLength();
        double[] initValues = Arrays.copyOf(values, initLength);

        CircularQueue<DataPoint> initResult = model.initialize(initValues);
        for (int i = 0; i < initLength; i++) {
            result.add(initResult.get(i).clone());
        }

        long t1 = System.nanoTime();
        for (int i = initLength; i < values.length; i++) {
            DataPoint point = model.decompose(values[i]);
            result.add(point.clone());
        }
        long t2 = System.nanoTime();

        CsvUtils.toCsv(result, output);
        evaluate(result, inputData);
        latencyUs = 1e-3 * (t2 - t1) / (values.length - initLength);
    }

    private static void evaluate(List<DataPoint> result, List<DataPoint> truth) {
        assert result.size() == truth.size();
        trendMAE = Utils.meanAbsoluteError(result.stream().map(DataPoint::getTrend).collect(Collectors.toList()),
            truth.stream().map(DataPoint::getTrend).collect(Collectors.toList()));
        seasonMAE = Utils.meanAbsoluteError(result.stream().map(DataPoint::getSeason).collect(Collectors.toList()),
            truth.stream().map(DataPoint::getSeason).collect(Collectors.toList()));
        residualMAE = Utils.meanAbsoluteError(result.stream().map(DataPoint::getResidual).collect(Collectors.toList()),
            truth.stream().map(DataPoint::getResidual).collect(Collectors.toList()));
    }

}
