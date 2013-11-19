package com.google.code.jneural;

import static org.junit.Assert.*;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import com.google.code.jmathematics.matrix.Matrix;
import com.google.code.jmathematics.matrix.longs.MutableLongMatrix;
import com.google.code.jmathematics.matrix.number.BigDecimalMutableMatrix;
import com.google.code.jneural.Perceptron.Training;

/**
 * Test input/output taken from the Wikipedia page
 * 
 * @see http://en.wikipedia.org/wiki/Perceptron
 */
public class PerceptronTest {

    private static final BigDecimal             LEARNING_RATE    = new BigDecimal("0.1");
    private static final BigDecimal             THRESHOLD        = new BigDecimal("0.5");
    private static final int                    NUM_TO_RECOGNISE = 4;
    private static final int                    NUM_ELEMENTS     = 3;
    private Perceptron<BigDecimalMutableMatrix> toTest;
    private BigDecimalMutableMatrix             weights = init0Weights();
    BigDecimalMutableMatrix[]                   questions = new BigDecimalMutableMatrix[NUM_TO_RECOGNISE];

    @Before
    public void setUp() {
        toTest = new Perceptron<BigDecimalMutableMatrix>(weights, THRESHOLD,  LEARNING_RATE);
    }

    private BigDecimalMutableMatrix init0Weights() {
        BigDecimalMutableMatrix weights = new BigDecimalMutableMatrix(1,
                NUM_ELEMENTS);
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            weights.set(i, 0, new BigDecimal(0));
        }
        return weights;
    }
    
    @Test
    public void network() {
        checkFactor("0", "0.5", "0.0");
        checkFactor("1", "0.0", "0.1");
        checkFactor("1", "0.6", "0.0");
    }

    private void checkFactor(String desired, String sum, String expected) {
        assertEquals(new BigDecimal(expected), toTest.calculateFactor(new BigDecimal(desired), new BigDecimal(sum)));
    }

    @Test
    public void testFirstIteration() {
        train();

        check("0.3", 0, 0);
        check("0.1", 1, 0);
        check("0.1", 2, 0);
    }

    @Test
    public void test8Iterations() {

        for (int i = 0 ; i < 8 ; i++) {
            train();
        }

        check("0.8", 0, 0);
        check("-0.2", 1, 0);
        check("-0.1", 2, 0);
    }

    private void train() {
        List<Training<BigDecimalMutableMatrix>> trainingSet = initNANDTrainingSet(1);
        toTest.teach(trainingSet);
    }
    
    private void train(int scaleFactor) {
        List<Training<BigDecimalMutableMatrix>> trainingSet = initNANDTrainingSet(scaleFactor);
        toTest.teach(trainingSet);
    }
    
    @Test
    public void shouldBeBelowThreshold() {
        assertTrue(toTest.isBelowThreshold(THRESHOLD.subtract(new BigDecimal("0.1"))));
        assertFalse(toTest.isBelowThreshold(THRESHOLD.add(new BigDecimal("0.1"))));
    }
    
    @Test
    public void shouldRecognize() {
        toTest = new Perceptron(weights, new BigDecimal("0.999"),  new BigDecimal("0.01"));
        
        int scaleFactor = 1;
        for (int i = 0 ; i < 1000 ; i++) {
            train(scaleFactor);
        }
        
        checkWithTolerance(scaleFactor, toTest.whatIs(questions[0]));
        checkWithTolerance(scaleFactor, toTest.whatIs(questions[1]));
        checkWithTolerance(scaleFactor, toTest.whatIs(questions[2]));
        checkWithTolerance(0,           toTest.whatIs(questions[3]));
    }
    
    private void checkWithTolerance(double expected, BigDecimal actual) {
        assertEquals(expected, actual.doubleValue(), 0.05d);
    }

    private List<Training<BigDecimalMutableMatrix>> initNANDTrainingSet(int scaleFactor) {
        List<Training<BigDecimalMutableMatrix>> trainingSet = new ArrayList<Perceptron.Training<BigDecimalMutableMatrix>>();
        
        int trueValue = 1 * scaleFactor;
        int bias      = 1;
        
        questions[0] = init(new int[] { bias, 0,           0 });
        questions[1] = init(new int[] { bias, 0,           trueValue });
        questions[2] = init(new int[] { bias, trueValue,   0 });
        questions[3] = init(new int[] { bias, trueValue,   trueValue });
        addToTraininSet(trainingSet, questions[0], trueValue);
        addToTraininSet(trainingSet, questions[1], trueValue);
        addToTraininSet(trainingSet, questions[2], trueValue);
        addToTraininSet(trainingSet, questions[3], 0);
        return trainingSet;
    }

    private void check(String expected, int x, int y) {
        assertEquals(new BigDecimal(expected), toTest.getWeights().get(x, y));
    }

    private void addToTraininSet(
            Collection<Training<BigDecimalMutableMatrix>>   set,
            BigDecimalMutableMatrix                         values, 
            int                                             expected) {
        set.add(new Training<BigDecimalMutableMatrix>(values,
                new BigDecimal(expected)));
    }

    private BigDecimalMutableMatrix init(int[] value) {
        BigDecimalMutableMatrix input = new BigDecimalMutableMatrix(3, 1);
        for (int i = 0; i < value.length; i++) {
            input.set(0, i, new BigDecimal(value[i]));
        }
        return input;
    }

}
