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
import edu.umass.cs.mallet.base.types.MatrixOps;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.Label;
import edu.umass.cs.mallet.base.types.LabelAlphabet;
import edu.umass.cs.mallet.base.types.FeatureVector;
import edu.umass.cs.mallet.base.types.Labeling;
import edu.umass.cs.mallet.base.types.LabelVector;
import edu.umass.cs.mallet.base.types.Vector;
import edu.umass.cs.mallet.base.types.FeatureSelection;
import edu.umass.cs.mallet.base.util.MalletLogger;
import edu.umass.cs.mallet.base.util.Maths;
import edu.umass.cs.mallet.base.maximize.Maximizable;
import edu.umass.cs.mallet.base.maximize.Maximizer;
import edu.umass.cs.mallet.base.maximize.LimitedMemoryBFGS;
import edu.umass.cs.mallet.base.pipe.Pipe;
import edu.umass.cs.mallet.base.util.CommandOption;
import java.util.logging.*;
import java.util.Random;

// Does not currently handle instances that are labeled with distributions
// instead of a single label.

public class MaxEntTrainer extends ClassifierTrainer implements Boostable //implements CommandOption.ListProviding
{
	private static Logger logger = MalletLogger.getLogger(MaxEntTrainer.class.getName());

	int numGetValueCalls = 0;
	int numGetValueGradientCalls = 0;
	int numIterations = 9999;
	
	// xxx Why does TestMaximizable fail when this variance is very small?
	static final double DEFAULT_GAUSSIAN_PRIOR_VARIANCE = 1.0;
	static final double DEFAULT_HYPERBOLIC_PRIOR_SLOPE = 0.2;
	static final double DEFAULT_HYPERBOLIC_PRIOR_SHARPNESS = 10.0;
	static final Class DEFAULT_MAXIMIZER_CLASS = LimitedMemoryBFGS.class;
	
	boolean usingHyperbolicPrior = false;
	double gaussianPriorVariance = DEFAULT_GAUSSIAN_PRIOR_VARIANCE;
	double hyperbolicPriorSlope = DEFAULT_HYPERBOLIC_PRIOR_SLOPE;
	double hyperbolicPriorSharpness = DEFAULT_HYPERBOLIC_PRIOR_SHARPNESS;
	Class maximizerClass = DEFAULT_MAXIMIZER_CLASS;

	static CommandOption.Boolean usingHyperbolicPriorOption =
	new CommandOption.Boolean (MaxEntTrainer.class, "useHyperbolicPrior", "true|false", false, false,
														 "Use hyperbolic (close to L1 penalty) prior over parameters", null);
	static CommandOption.Double gaussianPriorVarianceOption =
	new CommandOption.Double (MaxEntTrainer.class, "gaussianPriorVariance", "FLOAT", true, 10.0,
														"Variance of the gaussian prior over parameters", null);
	static CommandOption.Double hyperbolicPriorSlopeOption =
	new CommandOption.Double (MaxEntTrainer.class, "hyperbolicPriorSlope", "FLOAT", true, 0.2,
														"Slope of the (L1 penalty) hyperbolic prior over parameters", null);
	static CommandOption.Double hyperbolicPriorSharpnessOption =
	new CommandOption.Double (MaxEntTrainer.class, "hyperbolicPriorSharpness", "FLOAT", true, 10.0,
														"Sharpness of the (L1 penalty) hyperbolic prior over parameters", null);
	
	static final CommandOption.List commandOptions =
	new CommandOption.List (
		"Maximum Entropy Classifier",
		new CommandOption[] {
			usingHyperbolicPriorOption,
			gaussianPriorVarianceOption,
			hyperbolicPriorSlopeOption,
			hyperbolicPriorSharpnessOption,
		});

	public static CommandOption.List getCommandOptionList ()
	{
		return commandOptions;
	}

	/*
	public MaxEntTrainer(Maximizer.ByGradient maximizer)
	{
		this.maximizerByGradient = maximizer;
		this.usingHyperbolicPrior = false;
	}
	*/
	
	public MaxEntTrainer (CommandOption.List col)
	{
		this.usingHyperbolicPrior = usingHyperbolicPriorOption.value;
		this.gaussianPriorVariance = gaussianPriorVarianceOption.value;
		this.hyperbolicPriorSlope = hyperbolicPriorSlopeOption.value;
		this.hyperbolicPriorSharpness = hyperbolicPriorSharpnessOption.value;
	}

	public MaxEntTrainer ()
	{
		this (false);
	}

	public MaxEntTrainer (boolean useHyperbolicPrior)
	{
		this.usingHyperbolicPrior = useHyperbolicPrior;
	}
	
	public MaxEntTrainer (double gaussianPriorVariance)
	{
		this.usingHyperbolicPrior = false;
		this.gaussianPriorVariance = gaussianPriorVariance;
	}

	public MaxEntTrainer (double hyperbolicPriorSlope,
												double hyperbolicPriorSharpness)
	{
		this.usingHyperbolicPrior = true;
		this.hyperbolicPriorSlope = hyperbolicPriorSlope;
		this.hyperbolicPriorSharpness = hyperbolicPriorSharpness;
	}

	public Maximizable.ByGradient getMaximizableTrainer (InstanceList ilist)
	{
		return new MaximizableTrainer (ilist, null);
	}

	public MaxEntTrainer setNumIterations (int i)
	{
		numIterations = i;
		return this;
	}
	
	public Classifier train (InstanceList trainingSet,
													 InstanceList validationSet,
													 InstanceList testSet,
													 ClassifierEvaluating evaluator,
													 Classifier initialClassifier)
	{
		logger.fine ("trainingSet.size() = "+trainingSet.size());
		MaximizableTrainer mt = new MaximizableTrainer (trainingSet, (MaxEnt)initialClassifier);
		Maximizer.ByGradient maximizer = new LimitedMemoryBFGS();
		maximizer.maximize (mt);
		logger.info("MaxEnt ngetValueCalls:"+getValueCalls()+"\nMaxEnt ngetValueGradientCalls:"+getValueGradientCalls());
		boolean converged;

	 	for (int i = 0; i < numIterations; i++) {
			converged = maximizer.maximize (mt, 1);
			if (converged)
			 	break;
			else if (evaluator != null)
			 	if (!evaluator.evaluate (mt.getClassifier(), converged, i, mt.getValue(),
				 												 trainingSet, validationSet, testSet))
				 	break;
		}

		return mt.getClassifier ();
	}

	public int getValueGradientCalls() {return numGetValueGradientCalls;}
	public int getValueCalls() {return numGetValueCalls;}
//	public int getIterations() {return maximizerByGradient.getIterations();}
	
	public String toString()
	{
		return "MaxEntTrainer: "
			+ "("+maximizerClass.getName()+") "
			+ (usingHyperbolicPrior
				 ? ("hyperbolic slope="+hyperbolicPriorSlope+
						" sharpness="+hyperbolicPriorSharpness)
				 : ("gaussian variance="+gaussianPriorVariance));
	}

	

  // A private inner class that wraps up a MaxEnt classifier and its training data.
	// The result is a maximize.Maximizable function.
	private class MaximizableTrainer implements Maximizable.ByGradient
	{
		double[] parameters, constraints, cachedGradient;
		MaxEnt theClassifier;
		InstanceList trainingList;
		// The expectations are (temporarily) stored in the cachedGradient
		double cachedValue;
		boolean cachedValueStale;
		boolean cachedGradientStale;
		int numLabels;
		int numFeatures;
		int defaultFeatureIndex;						// just for clarity
		FeatureSelection featureSelection;
		FeatureSelection[] perLabelFeatureSelection;
		
		public MaximizableTrainer (){}

		public MaximizableTrainer (InstanceList ilist, MaxEnt theNewClassifier)
		{
			this.trainingList = ilist;
			Alphabet fd = ilist.getDataAlphabet();
			LabelAlphabet ld = (LabelAlphabet) ilist.getTargetAlphabet();
			// xxx ??? Roosting doesn't want it to be frozen
			//fd.stopGrowth();
			ld.stopGrowth();
			// Add one feature for the "default feature".
			this.numLabels = ld.size();
			this.numFeatures = fd.size() + 1;
			this.defaultFeatureIndex = numFeatures-1;
			this.parameters = new double [numLabels * numFeatures];
			this.constraints = new double [numLabels * numFeatures];
			this.cachedGradient = new double [numLabels * numFeatures];
			for (int i=0; i < numLabels; i++)
				for (int j=0; j < numFeatures; j++) {
					this.parameters[i*numFeatures + j] = 0.0;
					this.constraints[i*numFeatures + j] = 0.0;
					this.cachedGradient[i*numFeatures + j] = 0.0;
				}					
			this.featureSelection = ilist.getFeatureSelection();
			this.perLabelFeatureSelection = ilist.getPerLabelFeatureSelection();
			// Add the default feature index to the selection
			if (featureSelection != null)
				featureSelection.add (defaultFeatureIndex);
			if (perLabelFeatureSelection != null)
				for (int i = 0; i < perLabelFeatureSelection.length; i++)
					perLabelFeatureSelection[i].add (defaultFeatureIndex);
			// xxx Later change this to allow both to be set, but select which one to use by a boolean flag?
			assert (featureSelection == null || perLabelFeatureSelection == null);
			if (theNewClassifier != null) {
				this.theClassifier = theNewClassifier;
				assert (theNewClassifier.getInstancePipe() == ilist.getPipe());
			}
			else if (this.theClassifier == null) {
				this.theClassifier = new MaxEnt (ilist.getPipe(), parameters, featureSelection, perLabelFeatureSelection);
			}
			cachedValueStale = true;
			cachedGradientStale = true;

			// Initialize the constraints
			InstanceList.Iterator iter = trainingList.iterator ();
			logger.fine("Number of instances in training list = " + trainingList.size());
			while (iter.hasNext()) {
				double instanceWeight = iter.getInstanceWeight();
				Instance inst = iter.nextInstance();
				Labeling labeling = inst.getLabeling ();
				//logger.fine ("Instance "+ii+" labeling="+labeling);
				FeatureVector fv = (FeatureVector) inst.getData ();
				Alphabet fdict = fv.getAlphabet();
				assert (fv.getAlphabet() == fd);
				int li = labeling.getBestIndex();
				MatrixOps.rowPlusEquals (constraints, numFeatures, li, fv, instanceWeight);
				// For the default feature, whose weight is 1.0
				assert(!Double.isNaN(instanceWeight)) : "instanceWeight is NaN";
				assert(!Double.isNaN(li)) : "bestIndex is NaN";
				boolean hasNaN = false;
				for(int i = 0; i < fv.numLocations(); i++) {
					if(Double.isNaN(fv.valueAtLocation(i))) {
						logger.info("NaN for feature " + fdict.lookupObject(fv.indexAtLocation(i)).toString()); 
						hasNaN = true;
					}
				}
				if(hasNaN)
					logger.info("NaN in instance: " + inst.getName());

				constraints[li*numFeatures + defaultFeatureIndex] += 1.0 * instanceWeight;
			}
		}

		public MaxEnt getClassifier () { return theClassifier; }
		
		public double getParameter (int index) {
			return parameters[index];
		}
		
		public void setParameter (int index, double v) {
			cachedValueStale = true;
			cachedGradientStale = true;
			parameters[index] = v;
		}
		
		public int getNumParameters() {
			return parameters.length;
		}
		
		public void getParameters (double[] buff) {
			if (buff == null || buff.length != parameters.length)
				buff = new double [parameters.length];
			System.arraycopy (parameters, 0, buff, 0, parameters.length);
		}
	
		public void setParameters (double [] buff) {
			assert (buff != null);
			cachedValueStale = true;
			cachedGradientStale = true;
			if (buff.length != parameters.length)
				parameters = new double[buff.length];
			System.arraycopy (buff, 0, parameters, 0, buff.length);
		}

		// log probability of the training labels
		public double getValue ()
		{
			if (cachedValueStale) {
				numGetValueCalls++;
				cachedValue = 0;
				// We'll store the expectation values in "cachedGradient" for now
				cachedGradientStale = true;
				MatrixOps.setAll (cachedGradient, 0.0);
				// Incorporate likelihood of data
				double[] scores = new double[trainingList.getTargetAlphabet().size()];
				double value;
				InstanceList.Iterator iter = trainingList.iterator();
				int ii=0;
				while (iter.hasNext()) {
					ii++;
					double instanceWeight = iter.getInstanceWeight();
					Instance instance = iter.nextInstance();
					Labeling labeling = instance.getLabeling ();
					this.theClassifier.getClassificationScores (instance, scores);
					FeatureVector fv = (FeatureVector) instance.getData ();
					int li = labeling.getBestIndex();
					value = - (instanceWeight * Math.log (scores[li]));
					if(Double.isNaN(value)) {
						logger.fine ("MaxEntTrainer: Instance " + instance.getName() +
												 "has NaN value. log(scores)= " + Math.log(scores[li]) +
												 " scores = " + scores[li] + 
												 " has instance weight = " + instanceWeight);
						
					}
					if (Double.isInfinite(value)) {
						logger.warning ("Instance "+instance.getSource() + " has infinite value; skipping value and gradient");
						continue;
					}
					cachedValue += value;
					for (int si = 0; si < scores.length; si++) {
						if (scores[si] == 0) continue;
						assert (!Double.isInfinite(scores[si]));
						MatrixOps.rowPlusEquals (cachedGradient, numFeatures,
																		 si, fv, -instanceWeight * scores[si]);
						cachedGradient[numFeatures*si + defaultFeatureIndex] += -scores[si];
					}
				}
				//logger.info ("-Expectations:"); cachedGradient.print();
				// Incorporate prior on parameters
				if (usingHyperbolicPrior) {
					for (int li = 0; li < numLabels; li++)
						for (int fi = 0; fi < numFeatures; fi++)
							cachedValue += (hyperbolicPriorSlope / hyperbolicPriorSharpness
														 * Math.log (Maths.cosh (hyperbolicPriorSharpness * parameters[li *numFeatures + fi])));
				} else {
					for (int li = 0; li < numLabels; li++)
						for (int fi = 0; fi < numFeatures; fi++) {
							double param = parameters[li*numFeatures + fi];
							cachedValue += param * param / (2 * gaussianPriorVariance);
						}
				}
				cachedValue *= -1.0; // MAXIMIZE, NOT MINIMIZE
				cachedValueStale = false;
				logger.info ("Value (loglikelihood) = "+cachedValue);
			}
			return cachedValue;
		}

		public void getValueGradient (double [] buffer)
		{
			// Gradient is -(constraint - expectation - parameters/gaussianPriorVariance)
			if (cachedGradientStale) {
				numGetValueGradientCalls++;
				if (cachedValueStale)
					// This will fill in the cachedGradient with the "-expectation"
					getValue ();
				MatrixOps.plusEquals (cachedGradient, constraints);
				// Incorporate prior on parameters
				if (usingHyperbolicPrior) {
					throw new UnsupportedOperationException ("Hyperbolic prior not yet implemented.");
				} else {
					MatrixOps.plusEquals (cachedGradient, parameters,
																-1.0 / gaussianPriorVariance);
				}
				// xxx Show the feature with maximum gradient
				// Multiply by -1.0 because the gradient should point "up-hill",
				// which is actually away from the direction we want to parameters to go.
				MatrixOps.timesEquals (cachedGradient, -1.0);
				// A parameter may be set to -infinity by an external user.
				// We set gradient to 0 because the parameter's value can
				// never change anyway and it will mess up future calculations
				// on the matrix, such as norm().
				MatrixOps.substitute (cachedGradient, Double.NEGATIVE_INFINITY, 0.0);
				// Set to zero all the gradient dimensions that are not among the selected features
				if (perLabelFeatureSelection == null) {
					for (int labelIndex = 0; labelIndex < numLabels; labelIndex++)
						MatrixOps.rowSetAll (cachedGradient, numFeatures,
																 labelIndex, 0.0, featureSelection, false);
				} else {
					for (int labelIndex = 0; labelIndex < numLabels; labelIndex++)
						MatrixOps.rowSetAll (cachedGradient, numFeatures,
																 labelIndex, 0.0,
																 perLabelFeatureSelection[labelIndex], false);
				}
				cachedGradientStale = false;
			}
			if (buffer == null || buffer.length != parameters.length)
				buffer = new double[parameters.length];
			System.arraycopy (cachedGradient, 0, buffer, 0, cachedGradient.length);
		}
	}














}
