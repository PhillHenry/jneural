package com.google.code.jneural;

import java.math.BigDecimal;
import java.util.Collection;
import java.util.Set;

import com.google.code.jmathematics.matrix.Matrix;
import com.google.code.jmathematics.matrix.longs.LongMatrix;
import com.google.code.jmathematics.matrix.longs.MutableLongMatrix;
import com.google.code.jmathematics.matrix.number.BigDecimalMutableMatrix;
import com.google.code.jmathematics.matrix.number.NumberMatrix;

/**
 * @see http://en.wikipedia.org/wiki/Perceptron
 */
public class Perceptron<T extends Matrix<?>> {
	
    private final BigDecimal learningRate;
    private final BigDecimal threshold;

    public static class Training<BigDecimalMutableMatrix> {
		final BigDecimalMutableMatrix input;
		final BigDecimal desired;
        public Training(BigDecimalMutableMatrix x, BigDecimal d) {
            super();
            this.input = x;
            this.desired = d;
        }
	}
	
	private BigDecimalMutableMatrix weights;

	public Perceptron(BigDecimalMutableMatrix weights, BigDecimal threshold, BigDecimal learningRate) {
		super();
		this.weights = weights;
		this.learningRate = learningRate;
		this.threshold = threshold;
	}

	public void teach(Collection<Training<BigDecimalMutableMatrix>> trainingSet) {
		for (Training<BigDecimalMutableMatrix> lesson : trainingSet) {
			BigDecimal   weightsDotInput = (BigDecimal) weights.dotProduct(lesson.input);
			BigDecimal   desired         = lesson.desired;
			boolean      activated       = shouldBeActivated(weightsDotInput, desired);
            if (activated) {
			    BigDecimal factor            = calculateFactor(desired, weightsDotInput);
		        changeWeights(lesson, factor);
		    } 
		}
	}

    private boolean shouldBeActivated(BigDecimal weightsDotInput, BigDecimal desired) {
        return isBelowThreshold(weightsDotInput) || isBelowThreshold(desired);
    }

    private void changeWeights(Training<BigDecimalMutableMatrix> lesson, BigDecimal factor) {
        BigDecimalMutableMatrix factored            = lesson.input.scalar(factor);
        BigDecimalMutableMatrix factoredTransposed  = factored.transpose();
        weights                                     =  weights.add(factoredTransposed);
    }
	
	public BigDecimal whatIs(BigDecimalMutableMatrix values) {
	    return (BigDecimal) weights.dotProduct(values);
	}

    protected BigDecimal calculateFactor(BigDecimal desired, BigDecimal sum) {
        BigDecimal n    = isBelowThreshold(sum) ? new BigDecimal(0) : new BigDecimal(1);
        BigDecimal diff = desired.subtract(n);
        return diff.multiply(learningRate);
    }
    
    protected boolean isBelowThreshold(BigDecimal factor) {
        return factor.compareTo(threshold) <= 0;
    }
    
    BigDecimalMutableMatrix getWeights() {
        return weights;
    }

}
