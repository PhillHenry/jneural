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
	
    private final String learningRate = "0.1";
    private final BigDecimal threshold = new BigDecimal("0.5");

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

	public Perceptron(T weights) {
		super();
		this.weights = weights;
	}

	public void teach(Collection<Training<T>> trainingSet) {
		for (Training<T> lesson : trainingSet) {
		    System.out.println("Initial weights: " + weights + ", input: " + lesson.input.transpose());
			BigDecimal y = (BigDecimal) weights.dotProduct(lesson.input);
			BigDecimal d = lesson.desired;
			System.out.println("sum = " + y + ", desired = " + d);
			BigDecimal factor = calculateFactor(d, y);
            
			if (isBelowThreshold(y)) {
			    changeWeights(lesson, factor);
			} else {
			    if (lessThan(d, threshold)) {
			        changeWeights(lesson, factor);
			    } else {
			        System.out.println("not changing. d = " + d);
			    }
			}
			
		}
	}

    private void changeWeights(Training<T> lesson, BigDecimal factor) {
        T factored = (T) lesson.input.scalar(factor);
        T factoredTransposed = (T)factored.transpose();
        
        System.out.println("Adding: factor = " + factor + " to add = " + factoredTransposed);
        
        weights = (T) weights.add(factoredTransposed);
    }
	
	public BigDecimal whatIs(T values) {
	    return (BigDecimal) weights.dotProduct(values);
	}

    protected BigDecimal calculateFactor(BigDecimal z, BigDecimal sum) {
        BigDecimal n = isBelowThreshold(sum) ? new BigDecimal(0) : new BigDecimal(1);
        BigDecimal diff = z.subtract(n);
        return diff.multiply(new BigDecimal(learningRate));
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
