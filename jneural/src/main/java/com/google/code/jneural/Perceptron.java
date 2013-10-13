package com.google.code.jneural;

import java.math.BigDecimal;
import java.util.Collection;
import java.util.Set;

import com.google.code.jmathematics.matrix.Matrix;
import com.google.code.jmathematics.matrix.longs.LongMatrix;
import com.google.code.jmathematics.matrix.longs.MutableLongMatrix;
import com.google.code.jmathematics.matrix.number.BigDecimalMutableMatrix;

/**
 * @see http://en.wikipedia.org/wiki/Perceptron
 */
public class Perceptron<T extends BigDecimalMutableMatrix> {
	
    private final BigDecimal learningRate;
    private final BigDecimal threshold;

    public static class Training<T extends BigDecimalMutableMatrix> {
		final T input;
		final BigDecimal desired;
        public Training(T x, BigDecimal d) {
            super();
            this.input = x;
            this.desired = d;
        }
	}
	
	private T weights;

	public Perceptron(T weights, BigDecimal threshold, BigDecimal learningRate) {
		super();
		this.weights = weights;
		this.learningRate = learningRate;
		this.threshold = threshold;
	}

	public void teach(Collection<Training<T>> trainingSet) {
		for (Training<T> lesson : trainingSet) {
			BigDecimal y = (BigDecimal) weights.dotProduct(lesson.input);
			BigDecimal d = lesson.desired;
			BigDecimal factor = calculateFactor(d, y);
            
			if (isBelowThreshold(y)) {
			    changeWeights(lesson, factor);
			} else {
			    if (lessThan(d, threshold)) {
			        changeWeights(lesson, factor);
			    } else {
			    }
			}
			
		}
	}

    private void changeWeights(Training<T> lesson, BigDecimal factor) {
        T factored = (T) lesson.input.scalar(factor);
        T factoredTransposed = (T)factored.transpose();
        weights = (T) weights.add(factoredTransposed);
    }
	
	public BigDecimal whatIs(T values) {
	    return (BigDecimal) weights.dotProduct(values);
	}

    protected BigDecimal calculateFactor(BigDecimal z, BigDecimal sum) {
        BigDecimal n = isBelowThreshold(sum) ? new BigDecimal(0) : new BigDecimal(1);
        BigDecimal diff = z.subtract(n);
        return diff.multiply(learningRate);
    }
    
    protected boolean isBelowThreshold(BigDecimal factor) {
        return factor.compareTo(threshold) <= 0;
    }
    
    protected boolean lessThan(BigDecimal x, BigDecimal y) {
        return x.compareTo(y) < 0;
    }
	
    T getWeights() {
        return weights;
    }

}
