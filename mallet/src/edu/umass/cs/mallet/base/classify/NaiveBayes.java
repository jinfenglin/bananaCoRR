/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.classify;

import edu.umass.cs.mallet.base.classify.Classifier;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.FeatureVector;
import edu.umass.cs.mallet.base.types.Labeling;
import edu.umass.cs.mallet.base.types.LabelVector;
import edu.umass.cs.mallet.base.types.Multinomial;
import edu.umass.cs.mallet.base.pipe.Pipe;

public class NaiveBayes extends Classifier
{
	Multinomial.Logged prior;
	Multinomial.Logged[] p;

	public NaiveBayes (Pipe instancePipe,
										 Multinomial.Logged prior,
										 Multinomial.Logged[] classIndex2FeatureProb)
	{
		super (instancePipe);
		this.prior = prior;
		this.p = classIndex2FeatureProb;
	}

	private static Multinomial.Logged[] logMultinomials (Multinomial[] m)
	{
		Multinomial.Logged[] ml = new Multinomial.Logged[m.length];
		for (int i = 0; i < m.length; i++)
			ml[i] = new Multinomial.Logged (m[i]);
		return ml;
	}

	public NaiveBayes (Pipe dataPipe,
										 Multinomial prior,
										 Multinomial[] classIndex2FeatureProb)
	{
		this (dataPipe,
					new Multinomial.Logged (prior),
					logMultinomials (classIndex2FeatureProb));
	}

	public Classification classify (Instance instance)
	{
		int numClasses = getLabelAlphabet().size();
		double[] scores = new double[numClasses];
		FeatureVector fv = (FeatureVector) instance.getData (this.instancePipe);
		// Make sure the feature vector's feature dictionary matches
		// what we are expecting from our data pipe (and thus our notion
		// of feature probabilities.
		assert (instancePipe == null
						|| fv.getAlphabet () == instancePipe.getDataAlphabet ());
		int fvisize = fv.numLocations();

		prior.addLogProbabilities (scores);
		// Set the scores according to the feature weights and per-class probabilities
		for (int fvi = 0; fvi < fvisize; fvi++) {
			int fi = fv.indexAtLocation (fvi);
			for (int ci = 0; ci < numClasses; ci++)
				scores[ci] += fv.valueAtLocation (fvi) * p[ci].logProbability (fi);
		}

		// Get the scores in the range near zero, where exp() is more accurate
		double maxScore = Double.NEGATIVE_INFINITY;
		for (int ci = 0; ci < numClasses; ci++)
			if (scores[ci] > maxScore)
				maxScore = scores[ci];
		for (int ci = 0; ci < numClasses; ci++)
			scores[ci] -= maxScore;

		// Exponentiate and normalize
		double sum = 0;
		for (int ci = 0; ci < numClasses; ci++)
			sum += (scores[ci] = Math.exp (scores[ci]));
		for (int ci = 0; ci < numClasses; ci++)
			scores[ci] /= sum;

		// Create and return a Classification object
		return new Classification (instance, this,
															 new LabelVector (getLabelAlphabet(),
																								scores));
	}
	

}
