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
import edu.umass.cs.mallet.base.types.InstanceList;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.FeatureVector;
import edu.umass.cs.mallet.base.types.Labeling;
import edu.umass.cs.mallet.base.types.LabelVector;
import edu.umass.cs.mallet.base.types.Multinomial;
import edu.umass.cs.mallet.base.types.FeatureSelection;
import edu.umass.cs.mallet.base.pipe.Pipe;

public class NaiveBayesTrainer extends ClassifierTrainer implements Boostable
{
	Multinomial.Estimator featureEstimator = new Multinomial.LaplaceEstimator();
	Multinomial.Estimator priorEstimator = new Multinomial.LaplaceEstimator();

	public Multinomial.Estimator getFeatureMultinomialEstimator ()
	{
		return featureEstimator;
	}
		
	public void setFeatureMultinomialEstimator (Multinomial.Estimator me)
	{
		featureEstimator = me;
	}
		
	public Multinomial.Estimator getPriorMultinomialEstimator ()
	{
		return priorEstimator;
	}
		
	public void setPriorMultinomialEstimator (Multinomial.Estimator me)
	{
		priorEstimator = me;
	}

	public Classifier train (InstanceList trainingList,
													 InstanceList validationList,
													 InstanceList testSet,
													 ClassifierEvaluating evaluator,
													 Classifier initialClassifier)
	{
		FeatureSelection selectedFeatures = trainingList.getFeatureSelection();
		//if (selectedFeatures != null)
			// xxx Attend to FeatureSelection!!!
		//	throw new UnsupportedOperationException ("FeatureSelection not yet implemented.");
		trainingList.getDataAlphabet().stopGrowth();
		trainingList.getTargetAlphabet().stopGrowth();
		Pipe instancePipe = trainingList.getPipe ();
		Alphabet dict = (Alphabet) trainingList.getDataAlphabet ();
		int numLabels = trainingList.getTargetAlphabet().size();
		//System.out.println ("numLabels = "+numLabels);
		//System.out.println ("trainingList.size() = "+trainingList.size());
		Multinomial.Estimator[] me =
			new Multinomial.LaplaceEstimator[numLabels];
		for (int i = 0; i < numLabels; i++) {
			me[i] = (Multinomial.Estimator) featureEstimator.clone ();
			me[i].setAlphabet (dict);
		}
		Multinomial.Estimator pe =
			(Multinomial.Estimator) priorEstimator.clone ();

		InstanceList.Iterator iter = trainingList.iterator();
		while (iter.hasNext()) {
			double instanceWeight = iter.getInstanceWeight();
			Instance inst = iter.nextInstance();
			Labeling labeling = inst.getLabeling ();
			FeatureVector fv = (FeatureVector) inst.getData (instancePipe);
			for (int lpos = 0; lpos < labeling.numLocations(); lpos++) {
				int li = labeling.indexAtLocation (lpos);
				double labelWeight = labeling.valueAtLocation (lpos);
				if (labelWeight == 0) continue;
				//System.out.println ("NaiveBayesTrainer me.increment "+ labelWeight * instanceWeight);
				me[li].increment (fv, labelWeight * instanceWeight);
				// This relies on labelWeight summing to 1 over all labels
				pe.increment (li, labelWeight * instanceWeight);
			}
		}
		Multinomial[] m = new Multinomial[numLabels];
		for (int li = 0; li < numLabels; li++) {
			//me[li].print ();
			m[li] = me[li].estimate();
		}
		return new NaiveBayes (instancePipe, pe.estimate(), m);
	}

}
