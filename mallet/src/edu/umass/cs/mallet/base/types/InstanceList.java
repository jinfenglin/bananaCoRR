/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */



/**
	 A list of machine learning instances, typically used for training
	 or testing of a machine learning algorithm.

	 All of the instances in the list will have been passed through the
	 same Pipe.

   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.types;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import edu.umass.cs.mallet.base.types.Labeling;
import edu.umass.cs.mallet.base.pipe.Pipe;
import edu.umass.cs.mallet.base.pipe.PipeOutputAccumulator;
import edu.umass.cs.mallet.base.pipe.SerialPipes;
import edu.umass.cs.mallet.base.pipe.TokenSequence2FeatureSequence;
import edu.umass.cs.mallet.base.pipe.FeatureSequence2FeatureVector;
import edu.umass.cs.mallet.base.pipe.Target2Label;
import edu.umass.cs.mallet.base.pipe.iterator.PipeInputIterator;
import edu.umass.cs.mallet.base.pipe.iterator.RandomTokenSequenceIterator;
import edu.umass.cs.mallet.base.util.MalletLogger;
import edu.umass.cs.mallet.base.util.PropertyList;
import edu.umass.cs.mallet.base.util.Random;
import edu.umass.cs.mallet.base.util.DoubleList;
import edu.umass.cs.mallet.base.types.Instance;
import java.util.logging.*;
import java.io.*;

public class InstanceList implements Serializable, PipeOutputAccumulator
{
	private static Logger logger = MalletLogger.getLogger(InstanceList.class.getName());

	ArrayList instances;
	DoubleList instanceWeights = null;
	FeatureSelection featureSelection = null;
	FeatureSelection[] perLabelFeatureSelection = null;
	Pipe pipe;
	Alphabet dataVocab, targetVocab;
	Class dataClass = null;
	Class targetClass = null;
	
	public InstanceList (Pipe pipe, int capacity)
	{
		this.pipe = pipe;
		this.instances = new ArrayList (capacity);
	}

	public InstanceList (Pipe pipe)
	{
		this (pipe, 10);
	}

	/** Create an InstanceList in which the vocabulary objects do not
			come from a pipe.

			Used in those infrequent circumstances when the InstanceList has
			no pipe, and objects containing vocabularies are entered
			directly into the InstanceList; for example, the creation of a
			random InstanceList using Dirichlets and Multinomials.
	*/
	public InstanceList (Alphabet dataVocab, Alphabet targetVocab)
	{
		this (null, 10);
		this.dataVocab = dataVocab;
		this.targetVocab = targetVocab;
	}

	private static class NotYetSetPipe extends Pipe	{
		public Instance pipe (Instance carrier)	{
			throw new UnsupportedOperationException (
				"The InstanceList has yet to have its pipe set; "+
				"this could happen by calling InstanceList.add(InstanceList)");
		}
	}
	static final Pipe notYetSetPipe = new NotYetSetPipe();

	public InstanceList ()
	{
		this (notYetSetPipe);
	}

	// Return a InstanceList consisting of randomly-generated FeatureVectors
	// xxx Perhaps split these out into a utility class
	public InstanceList (Random r,
											 // the generator of all random-ness used here
											 Dirichlet classCentroidDistribution,
											 // includes a Alphabet
											 double classCentroidAverageAlphaMean,
											 // Gaussian mean on the sum of alphas
											 double classCentroidAverageAlphaVariance,
											 // Gaussian variance on the sum of alphas
											 double featureVectorSizePoissonLambda,
											 double classInstanceCountPoissonLambda,
											 String[] classNames)
	{
		this (new SerialPipes (new Pipe[]	{
			new TokenSequence2FeatureSequence (),
			new FeatureSequence2FeatureVector (),
			new Target2Label()}));
		//classCentroidDistribution.print();
		PipeInputIterator iter = new RandomTokenSequenceIterator (
			r, classCentroidDistribution,
			classCentroidAverageAlphaMean, classCentroidAverageAlphaVariance,
			featureVectorSizePoissonLambda, classInstanceCountPoissonLambda,
			classNames);
		this.add (iter);
	}

	private static Alphabet dictOfSize (int size)
	{
		Alphabet ret = new Alphabet ();
		for (int i = 0; i < size; i++)
			ret.lookupIndex ("feature"+i);
		return ret;
	}

	private static String[] classNamesOfSize (int size)
	{
		String[] ret = new String[size];
		for (int i = 0; i < size; i++)
			ret[i] = "class"+i;
		return ret;
	}

	public InstanceList (Random r, Alphabet vocab, String[] classNames,
											 int meanInstancesPerLabel)
	{
		this (r, new Dirichlet(vocab, 2.0),
					30, 0,
					10, meanInstancesPerLabel, classNames);
	}
		
	public InstanceList (Random r, int vocabSize, int numClasses)
	{
		this (r, new Dirichlet(dictOfSize(vocabSize), 2.0),
					30, 0,
					10, 20, classNamesOfSize(numClasses));
	}

	public InstanceList shallowClone ()
	{
		InstanceList ret = new InstanceList (pipe, instances.size());
		for (int i = 0; i < instances.size(); i++)
			ret.add ((Instance)instances.get(i));
		if (instanceWeights == null)
			ret.instanceWeights = null;
		else
			ret.instanceWeights = instanceWeights.cloneDoubleList();
		return ret;
	}

	public InstanceList cloneEmpty ()
	{
		InstanceList ret = new InstanceList (pipe);
		ret.instanceWeights = instanceWeights == null ? null : (DoubleList) instanceWeights.clone();
		// xxx Should the featureSelection and perLabel... be cloned?
		// Note that RoostingTrainer currently depends on not cloning its splitting.
		ret.featureSelection = this.featureSelection;
		ret.perLabelFeatureSelection = this.perLabelFeatureSelection;
		ret.dataClass = this.dataClass;
		ret.targetClass = this.targetClass;
		return ret;
	}
		
	public InstanceList[] split (java.util.Random r, double[] proportions)
	{
		double[] maxind = new double[proportions.length];
		System.arraycopy (proportions, 0, maxind, 0, proportions.length);
		InstanceList[] ret = new InstanceList[maxind.length];
		ArrayList shuffled = (ArrayList) this.instances.clone();
		Collections.shuffle (shuffled, r);
		DenseVector.normalize(maxind);
		//Vector.print (maxind);
		// Fill maxind[] with the highest instance index that should go in
		// each corresponding returned InstanceList.
		for (int i = 0; i < maxind.length; i++) {
			// xxx Is it dangerous to share the featureSelection that comes with cloning?
			ret[i] = this.cloneEmpty();
			if (i > 0)
				maxind[i] += maxind[i-1];
		}
		for (int i = 0; i < maxind.length; i++)
			maxind[i] = Math.rint (maxind[i] * shuffled.size());
		//Vector.print (maxind);
		int j = 0;
		// This gives a slight bias toward putting an extra instance in the last InstanceList.
		for (int i = 0; i < shuffled.size(); i++) {
			while (i >= maxind[j])
				j++;
			ret[j].instances.add (shuffled.get(i));
		}
		return ret;
	}

	public InstanceList[] split (double[] proportions)
	{
		return split (new java.util.Random(), proportions);
	}

	public InstanceList[] splitInOrder (double[] proportions)
	{
		double[] maxind = new double[proportions.length];
		System.arraycopy (proportions, 0, maxind, 0, proportions.length);
		InstanceList[] ret = new InstanceList[proportions.length];
		DenseVector.normalize(maxind);
		// Fill maxind[] with the highest instance index that should go in
		// each corresponding returned InstanceList.
		for (int i = 0; i < maxind.length; i++) {
			// xxx Is it dangerous to share the featureSelection that comes with cloning?
			ret[i] = this.cloneEmpty();
			if (i > 0)
				maxind[i] += maxind[i-1];
		}
		for (int i = 0; i < maxind.length; i++)
			maxind[i] = Math.rint (maxind[i] * this.size());
		int j = 0;
		// This gives a slight bias toward putting an extra instance in the last InstanceList.
		for (int i = 0; i < this.size(); i++) {
			while (i >= maxind[j])
				j++;
			ret[j].instances.add (this.get(i));
		}
		return ret;
	}

	public InstanceList[] splitByModulo (int m)
	{
		InstanceList[] ret = new InstanceList[2];
		ret[0] = this.cloneEmpty();
		ret[1] = this.cloneEmpty();
		for (int i = 0; i < this.size(); i++) {
			if (i % m == 0)
				ret[0].instances.add (this.get(i));
			else
				ret[1].instances.add (this.get(i));
		}
		return ret;
	}
	
	public InstanceList sampleWithReplacement (java.util.Random r, int numSamples)
	{
		InstanceList ret = this.cloneEmpty();
		for (int i = 0; i < numSamples; i++)
			ret.instances.add (this.getInstance(r.nextInt(instances.size())));
		return ret;
	}
		
	public Instance getInstance (int index)
	{
		return (Instance) instances.get (index);
	}

	public double getInstanceWeight (int index)
	{
		if (instanceWeights == null)
			return 1.0;
		else
			return instanceWeights.get(index);
	}

	public void setInstanceWeight (int index, double weight)
	{
		//System.out.println ("setInstanceWeight index="+index+" weight="+weight);
		if (weight != getInstanceWeight(index)) {
			if (instanceWeights == null)
				instanceWeights = new DoubleList (instances.size(), 1.0);
			instanceWeights.set (index, weight);
		}
	}
		
	public void setFeatureSelection (FeatureSelection selectedFeatures)
	{
		if (selectedFeatures != null
				&& selectedFeatures.getAlphabet() != null  // xxx We allow a null vocabulary here?  See CRF3.java
				&& selectedFeatures.getAlphabet() != getDataAlphabet())
			throw new IllegalArgumentException ("Vocabularies do not match");
		featureSelection = selectedFeatures;
	}
		
	public FeatureSelection getFeatureSelection ()
	{
		return featureSelection;
	}

	public void setPerLabelFeatureSelection (FeatureSelection[] selectedFeatures)
	{
		if (selectedFeatures != null) {
			for (int i = 0; i < selectedFeatures.length; i++)
				if (selectedFeatures[i].getAlphabet() != getDataAlphabet())
					throw new IllegalArgumentException ("Vocabularies do not match");
		}
		perLabelFeatureSelection = selectedFeatures;
	}
		
	public FeatureSelection[] getPerLabelFeatureSelection ()
	{
		return perLabelFeatureSelection;
	}

	/** Set the "target" field to null in all instances.  This makes unlabeled data. */
	public void removeTargets()
	{
		for (int i = 0; i < instances.size(); i++)
			getInstance(i).setTarget (null);
	}

	/** Set the "source" field to null in all instances.  This will often save memory when
			the raw data had been  placed in that field. */
	public void removeSources()
	{
		for (int i = 0; i < instances.size(); i++)
			getInstance(i).clearSource();
	}
		
	public Object get (int index)
	{
		return instances.get (index);
	}

	public static InstanceList load (File file)
	{
		try {
			ObjectInputStream ois = new ObjectInputStream (new FileInputStream (file));
			InstanceList ilist = (InstanceList) ois.readObject();
			ois.close();
			return ilist;
		} catch (Exception e) {
			e.printStackTrace();
			throw new IllegalArgumentException ("Couldn't read InstanceList from file "+file);
		}
	}

	// Serialization of InstanceList

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 1;
		
	private void writeObject (ObjectOutputStream out) throws IOException {
		int i, size;
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject(instances);
		out.writeObject(instanceWeights);
		out.writeObject(pipe);
	}
		
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int i, size;
		int version = in.readInt ();
		instances = (ArrayList) in.readObject();
		instanceWeights = (DoubleList) in.readObject();
		pipe = (Pipe) in.readObject();
	}

	// added - culotta@cs.umass.edu
	/**
		 CrossValidationIterator allows iterating over pairs of
		 InstanceLists, where each pair is split into training/testing
		 based on nfolds.
	 */	
	public class CrossValidationIterator implements java.util.Iterator, Serializable
	{
		int nfolds;
		InstanceList[] folds;
		int index;

		/**
			 @param _nfolds number of folds to split InstanceList into
			 @param seed seed for random number used to split InstanceList
		 */
		public CrossValidationIterator (int _nfolds, int seed)
		{			
			assert (_nfolds > 0) : "nfolds: " + nfolds;
			this.nfolds = _nfolds;
			this.index = 0;
			folds = new InstanceList[_nfolds];		 
			double fraction = (double) 1 / _nfolds;
			double[] proportions = new double[_nfolds];
			for (int i=0; i < _nfolds; i++) 
				proportions[i] = fraction;
			folds = split (new java.util.Random (seed), proportions);
			
		}

		public CrossValidationIterator (int _nfolds) {
			this (_nfolds, 1);
		}

		public boolean hasNext () { return index < nfolds; }

		/**
			 Returns the next training/testing split.
			 @return InstanceList[0]: larger split (training)
			         InstanceList[1]: smaller split (testing)
		 */
		public InstanceList[] nextSplit () {
			InstanceList[] ret = new InstanceList[2];
			ret[0] = new InstanceList (pipe);
			for (int i=0; i < folds.length; i++) {
				if (i==index)
					continue;
				InstanceList.Iterator iter = folds[i].iterator();
				while (iter.hasNext()) 
					ret[0].add (iter.nextInstance());									
			}
			ret[1] = folds[index].shallowClone();
			index++;
			return ret;
		}
		public Object next () { return nextSplit(); }		
		public void remove () { throw new UnsupportedOperationException(); }
	}

	
	public class Iterator implements java.util.Iterator, Serializable
	{
		int index;
		public Iterator () { this.index = 0; }
		public boolean hasNext () { return index < instances.size(); }
		public Instance nextInstance () { return (Instance)instances.get(index++); }
		public double getInstanceWeight () { return instanceWeights == null ? 1.0 : instanceWeights.get(index); }
		public Object next () { return nextInstance(); }
		public void remove () { throw new UnsupportedOperationException(); }

		// Serialization of InstanceListIterator
		private static final long serialVersionUID = 1;
		private static final int CURRENT_SERIAL_VERSION = 0;
		private void writeObject (ObjectOutputStream out) throws IOException {
			out.writeInt (CURRENT_SERIAL_VERSION);
			out.writeInt(index);
		}
		private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
			int version = in.readInt ();
			index = in.readInt();
		}
	}

	public int size ()
	{
		return instances.size();
	}

	public Class getDataClass ()
	{
		if (instances.size() == 0)
			return null;
		else
			return getInstance(0).getData().getClass();
	}

	public Pipe getPipe ()
	{
		return pipe;
	}

	public Alphabet getDataAlphabet ()
	{
		if (dataVocab == null && pipe != null) {
			dataVocab = pipe.getDataAlphabet ();
		}
		assert (pipe == null
						|| pipe.getDataAlphabet () == null
						|| pipe.getDataAlphabet () == dataVocab);
		return dataVocab;
	}

	public Alphabet getTargetAlphabet ()
	{
		if (targetVocab == null && pipe != null) {
			targetVocab = pipe.getTargetAlphabet ();
		}
		assert (pipe == null
						|| pipe.getTargetAlphabet () == null
						|| pipe.getTargetAlphabet () == targetVocab);
		return targetVocab;
	}
	
	public LabelVector targetLabelDistribution ()
	{
		if (instances.size() == 0) return null;
		if (!(getInstance(0).getTarget() instanceof Labeling))
			throw new IllegalStateException ("Target is not a labeling.");
		double[] counts = new double[getTargetAlphabet().size()];
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = (Instance) instances.get(i);
			Labeling l = (Labeling) instance.getTarget();
			l.addTo (counts, getInstanceWeight(i));
		}
		return new LabelVector ((LabelAlphabet)getTargetAlphabet(), counts);
	}


	// For PipeOutputAccumulator interface
	public void pipeOutputAccumulate (Instance carrier,	Pipe iteratedPipe)
	{
		// xxx ??? assert (iteratedPipe == pipe);
		// The assertion above won't be true when using IteratedPipe...
		//logger.fine ("pipeOutputAccumulate target="+target);
		// These various add() methods below will make sure that the Pipes match appropriately
		if (carrier.getData() instanceof InstanceList)
			add ((InstanceList)carrier.getData());
		else if (carrier.getData() instanceof PipeInputIterator)
			add ((PipeInputIterator)carrier.getData());
		else if (carrier.getData() instanceof Instance)
			add ((Instance)carrier.getData());
		else {
			if (pipe == notYetSetPipe)
				pipe = iteratedPipe;
			//System.out.println ("Instance.pipeOuputAccumulate carrier.getSource()="+carrier.getSource());
			add (new Instance (carrier.getData(), carrier.getTarget(), carrier.name, carrier.getSource(), iteratedPipe));
		}
	}

	public PipeOutputAccumulator clonePipeOutputAccumulator ()
	{
		return (PipeOutputAccumulator)shallowClone();
	}

	public Iterator iterator ()
	{
		return new Iterator();
	}

	public CrossValidationIterator crossValidationIterator (int nfolds, int seed)
	{
		return new CrossValidationIterator(nfolds, seed);
	}

	public CrossValidationIterator crossValidationIterator (int nfolds)
	{
		return crossValidationIterator(nfolds);
	}

	public void add (PipeInputIterator pi)
	{
		while (pi.hasNext()) {
			Instance carrier = pi.nextInstance();
			// xxx Perhaps try to arrange this so that a new Instance does not have to allocated.
			add (new Instance (carrier.getData(), carrier.getTarget(), carrier.name, carrier.getSource(), this.pipe));
		}
	}

	public void add (InstanceList ilist)
	{
		if (ilist.pipe == pipe) {
			Iterator iter = ilist.iterator();
			while (iter.hasNext())
				add(iter.nextInstance ());
		} else if (pipe == notYetSetPipe) {
			// This InstanceList doesn't have a pipe defined, but "ilist" does.
			// Take ilist's pipe as our own, and add its Instances directly.
			if (this.instances.size() > 0)
				// We don't want to have some instances in this list passed through
				// no pipe, and others passing through the new pipe.
				throw new IllegalArgumentException (
					"Trying to set this InstanceList's pipe, but it already has instances.");
			this.pipe = ilist.pipe;
			Iterator iter = ilist.iterator();
			while (iter.hasNext())
				add (iter.nextInstance());
		} else if (ilist.pipe == null) {
			// Treat the data from the instances in ilist as inputData for our pipe.
			Iterator iter = ilist.iterator();
			while (iter.hasNext())
				add (iter.nextInstance ());
		}	else
			// xxx Another thing to consider is that we could take the
			// ilist instances that were passed through its pipe, and pass
			// them through this InstanceList's pipe.  This seems
			// dangerous, though, and this InstanceList's pipe doesn't
			// reflect all processing.
			throw new IllegalArgumentException (
				"Instances to be added to a InstanceList cannot already have been piped, "
				+"unless the pipes are equal, or one of the pipes is null.");
	}

	public boolean add (Object data, Object target, Object name, Object source, double instanceWeight)
	{
		return add (new Instance (data, target, name, source, pipe), instanceWeight);
	}

	public boolean add (Object data, Object target, Object name, Object source)
	{
		return add (data, target, name, source, 1.0);
	}
		
	public boolean add (Instance instance)
	{
		if (pipe == notYetSetPipe)
			pipe = instance.getPipe();
		else if (instance.getPipe() != pipe)
			// Making sure that the Instance has the same pipe as us.
			// xxx This also is a good time check that the constituent data is
			// of a consistent type?
			throw new IllegalArgumentException ("pipes don't match: instance: "+
																					instance.getPipe()+" Instance.list: "+
																					this.pipe);
		if (dataClass == null) {
			dataClass = instance.data.getClass();
      if (pipe.isTargetProcessing())
        targetClass = instance.target.getClass();
		}
		return instances.add (instance);
	}
		
	public boolean add (Instance instance, double instanceWeight)
	{
		boolean ret = instances.add (instance);
		if (instanceWeight != 1.0 || instanceWeights != null) {
			if (instanceWeights == null)
				instanceWeights = new DoubleList (instances.size(), 1.0);
			instanceWeights.add (instanceWeight);
		}
		return ret;
	}


	/*
	// xxx Does this really belong here?
	// How would we match this result if we read more test instances with a pipe?
	public void trimFeaturesByCount (int minCount)
	{
		Alphabet oldv = pipe.getDataAlphabet ();
		Alphabet newv = new Alphabet ();
		int[] counts = new int[this.size()];

		// Get counts
		for (int i = 0; i < this.size(); i++) {
			Object data = this.getInstance(i).getData ();
			if (data instanceof FeatureVectorSequence) {
				FeatureVectorSequence fvs = (FeatureVectorSequence) data;
				for (int j = 0; j < fvs.size(); j++) {
					FeatureVector fv = fvs.getFeatureVector (j);
					for (int k = fv.numLocations()-1; k >= 0; k--)
						counts[fv.indexAtLocation(k)]++;
				}
			} else {
				throw new IllegalArgumentException ("Doesn't handle data of type "+data.getClass().getName());
			}
		}

		// Substitute in the new Alphabet
		dataDict = newv;
		// xxx Do this for the pipe too!
		// xxx Do it with a new method Pipe.setDataAlphabet ();

		// Build replacement FeatureVectorSequences with pruned features and new Alphabet
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = this.getInstance(i);
			Object data = instance.getData ();
			if (data instanceof FeatureVectorSequence) {
				FeatureVectorSequence fvs = (FeatureVectorSequence) data;
				FeatureVector[] fva = new FeatureVector[fvs.size()];
				for (int j = 0; j < fvs.size(); j++) {
					FeatureVector fv = fvs.getFeatureVector (j);
					AugmentableFeatureVector afv = new AugmentableFeatureVector (newv, fv.isBinary());
					for (int k = fv.numLocations()-1; k >= 0; k--)
						if (counts[fv.indexAtLocation(k)] >= minCount)
							afv.add (fv.indexAtLocation(k), fv.valueAtLocation(k));
					fva[j] = fv instanceof AugmentableFeatureVector ? afv : afv.toFeatureVector();
				}
				instance.data = new FeatureVectorSequence (fva);
			} else {
				throw new IllegalArgumentException ("Doesn't handle data of type "+data.getClass().getName());
			}
		}
	}
	*/



	// xxx Perhaps make public?
	// A collection of instances without random access (perhaps because backed on disk)
	protected interface Stream extends PipeOutputAccumulator
	{
		public Iterator iterator ();
		// Returns -1 for an "infinite" stream
		public int size();
		public Pipe getInstancePipe ();
		public Alphabet getTargetAlphabet ();
		public Alphabet getDataAlphabet ();
	}
	

}
