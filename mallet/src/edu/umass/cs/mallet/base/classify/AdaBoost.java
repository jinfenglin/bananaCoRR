/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 AdaBoost

	 Robert E. Schapire.
	 "The boosting approach to machine learning: An overview."
	 In MSRI Workshop on Nonlinear Estimation and Classification, 2002. 
	 http://www.research.att.com/~schapire/cgi-bin/uncompress-papers/msri.ps

	 Warning this class has not been tested at all!
	 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.classify;

import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.types.*;

public class AdaBoost extends Classifier
{
	Classifier[] weakClassifiers;
	double[] alphas;

	public AdaBoost (Pipe instancePipe, Classifier[] weakClassifiers, double[] alphas)
	{
		super (instancePipe);
		this.weakClassifiers = weakClassifiers;
		this.alphas = alphas;
	}

	public Classification classify (Instance inst)
	{
		int numClasses = getLabelAlphabet().size();
		double[] scores = new double[numClasses];
		int bestIndex;
		double sum = 0;
		// Gather scores of all weakClassifiers
		for (int round = 0; round < weakClassifiers.length; round++) {
			bestIndex = weakClassifiers[round].classify(inst).getLabeling().getBestIndex();
			scores[bestIndex] += alphas[round];
			sum += scores[bestIndex];
		}
		// Normalize the scores
		for (int i = 0; i < scores.length; i++)
			scores[i] /= sum;
		return new Classification (inst, this, new LabelVector (getLabelAlphabet(), scores));
	}

}
