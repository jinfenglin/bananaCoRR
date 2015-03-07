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

import edu.umass.cs.mallet.base.types.*;

public class AdaBoostTrainer extends ClassifierTrainer
{
	ClassifierTrainer weakLearner;
	int numRounds;
	
	public AdaBoostTrainer (ClassifierTrainer weakLearner, int numRounds)
	{
		if (!(weakLearner instanceof Boostable))
			throw new IllegalArgumentException ("weakLearner is not Boostable");
		this.weakLearner = weakLearner;
		this.numRounds = numRounds;
	}

	public AdaBoostTrainer (ClassifierTrainer weakLearner)
	{
		this (weakLearner, 100);
	}
	
	public Classifier train (InstanceList trainingList,
													 InstanceList validationList,
													 InstanceList testSet,
													 ClassifierEvaluating evaluator,
													 Classifier initialClassifier)
	{
		FeatureSelection selectedFeatures = trainingList.getFeatureSelection();
		boolean[] correct = new boolean[trainingList.size()];
		//assert (trainingList.getTargetAlphabet().size() == 2);
		int numClasses = trainingList.getTargetAlphabet().size();
		if (numClasses != 2)
			System.err.println ("AdaBoostTrainer.train: WARNING: more than two classes");
		Classifier[] weakLearners = new Classifier[numRounds];
		double[] alphas = new double[numRounds];
		// Set weights to be uniform
		double w = 1.0 / trainingList.size();
		for (int i = 0; i < trainingList.size(); i++)
			trainingList.setInstanceWeight (i, w);

		double sum, err;
		for (int round = 0; round < numRounds; round++) {
			sum = err = 0;
			weakLearners[round] = weakLearner.train (trainingList, validationList);
			for (int i = 0; i < trainingList.size(); i++) {
				Instance inst = trainingList.getInstance(i);
				correct[i] = weakLearners[round].classify(inst).bestLabelIsCorrect();
				if (!correct[i])
					err += trainingList.getInstanceWeight(i);
			}
			if (err > 1.0 / numClasses)
				throw new IllegalStateException ("weakLearner failed to do better than guessing; numClasses="+numClasses+" err="+err);
			// xxx This formula is really designed for binary classifiers that don't
			// give a confidence score.  We should do something smarter here.
			// See paper by Schapire & Singer.
			alphas[round] = 0.5 * Math.log ((1-err)/err);
			for (int i = 0; i < trainingList.size(); i++) {
				Instance inst = trainingList.getInstance(i);
				w = trainingList.getInstanceWeight(i);
				// xxx Attend to confidence scores here?
				if (correct[i])
					w *= Math.exp (-alphas[round]);
				else
					w *= Math.exp (alphas[round]);
				trainingList.setInstanceWeight (i, w);
				sum += w;
			}
			// Normalize the D_t's
			for (int i = 0; i < trainingList.size(); i++) {
				trainingList.setInstanceWeight (i, trainingList.getInstanceWeight(i) / sum);
			}
		}
		return new AdaBoost (trainingList.getPipe(), weakLearners, alphas);
	}
	
}
