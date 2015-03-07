/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.types;

import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.FeatureSequence;
import edu.umass.cs.mallet.base.types.FeatureVector;
import edu.umass.cs.mallet.base.util.Random;

public class Multinomial extends FeatureVector
{
	//	protected Multinomial () { }

	// "size" is the number of entries in "probabilities" that have valid values in them;
	// note that the dictionary (and thus the resulting multinomial) may be bigger than size
	// if the dictionary is shared with multiple estimators, and the dictionary grew
	// due to another estimator.
	private static double[] getValues (double[] probabilities, Alphabet dictionary,
																		 int size, boolean copy, boolean checkSum)
	{
		double[] values;
		assert (dictionary == null || dictionary.size() >= size);
		// No, not necessarily true; see comment above.
		//assert (dictionary == null || dictionary.size() == size);
		//assert (probabilities.length == size);
		// xxx Consider always copying, so that we are assured that we
		// always have a real probability distribution.
		if (copy) {
			values = new double[dictionary==null ? size : dictionary.size()];
			System.arraycopy (probabilities, 0, values, 0, size);
		} else {
			assert (dictionary == null || dictionary.size() == probabilities.length);
			values = probabilities;
		}
		if (checkSum) {
			// Check that we have a true probability distribution
			double sum = 0;
			for (int i = 0; i < values.length; i++)
				sum += values[i];
			if (Math.abs (sum - 1.0) > 0.9999)
				throw new IllegalArgumentException ("Probabilities sum to "
																						+ sum + ", not to one.");
		}
		return values;
	}

	protected Multinomial (double[] probabilities, Alphabet dictionary,
												 int size, boolean copy, boolean checkSum)
	{
		super (dictionary, getValues(probabilities, dictionary, size, copy, checkSum));
	}

	public Multinomial (double[] probabilities, Alphabet dictionary)
	{
		this (probabilities, dictionary, dictionary.size(), true, true);
	}

	public Multinomial (double[] probabilities, int size)
	{
		this (probabilities, null, size, true, true);
	}

	public Multinomial (double[] probabilities)
	{
		this (probabilities, null, probabilities.length, true, true);
	}


	public int size ()
	{
		return values.length;
	}

	public double probability (int featureIndex)
	{
		return values[featureIndex];
	}

	public double probability (Object key)
	{
		if (dictionary == null)
			throw new IllegalStateException ("This Multinomial has no dictionary.");
		return probability (dictionary.lookupIndex (key));
	}

	public double logProbability (int featureIndex)
	{
		return Math.log(values[featureIndex]);
	}

	public double logProbability (Object key)
	{
		if (dictionary == null)
			throw new IllegalStateException ("This Multinomial has no dictionary.");
		return logProbability (dictionary.lookupIndex (key));
	}

	public Alphabet getAlphabet ()
	{
		return dictionary;
	}

	public void addProbabilitiesTo (double[] vector)
	{
		for (int i = 0; i < values.length; i++)
			vector[i] += values[i];
	}


	public int randomIndex (Random r)
	{
		double f = r.nextUniform();
		double sum = 0;
		int i;
		for (i = 0; i < values.length; i++) {
			sum += values[i];
			//System.out.print (" sum="+sum);
			if (sum >= f)
				break;
		}
		//if (sum < f) throw new IllegalStateException
		//System.out.println ("i = "+i+", f = "+f+", sum = "+sum);
		assert (sum >= f);
		return i;
	}

	public Object randomObject (Random r)
	{
		if (dictionary == null)
			throw new IllegalStateException ("This Multinomial has no dictionary.");
		return dictionary.lookupObject (randomIndex (r));
	}

	public FeatureSequence randomFeatureSequence (Random r, int length)
	{
		if (! (dictionary instanceof Alphabet))
			throw new UnsupportedOperationException
				("Multinomial's dictionary much be a Alphabet");
		FeatureSequence fs = new FeatureSequence ((Alphabet)dictionary, length);
		while (length-- > 0)
			fs.add (randomIndex (r));
		return fs;
	}

	// "size" is the number of 1.0-weight features in the feature vector
	public FeatureVector randomFeatureVector (Random r, int size)
	{
		return new FeatureVector (randomFeatureSequence (r, size));
	}

	
	/** A Multinomial in which values[] caches Math.log(probability[fi]) instead of
			probability[fi] */
	public static class Logged extends Multinomial
	{
		public Logged (double[] probabilities, Alphabet dictionary,
									 int size, boolean areLoggedAlready)
		{
			super (probabilities, dictionary, size, true, !areLoggedAlready);
			assert (dictionary == null || dictionary.size() == size);
			if (!areLoggedAlready)
				for (int i = 0; i < size; i++)
					values[i] = Math.log (values[i]);
		}

		public Logged (double[] probabilities, Alphabet dictionary,
									 boolean areLoggedAlready)
		{
			this (probabilities, dictionary,
						(dictionary == null ? probabilities.length : dictionary.size()),
						areLoggedAlready);
		}

		public Logged (double[] probabilities, Alphabet dictionary, int size)
		{
			this (probabilities, dictionary, size, false);
		}

		public Logged (double[] probabilities, Alphabet dictionary)
		{
			this (probabilities, dictionary, dictionary.size(), false);
		}
		
		public Logged (Multinomial m)
		{
			this (m.values, m.dictionary, false);
		}

		public Logged (double[] probabilities)
		{
			this (probabilities, null, false);
		}

		public double probability (int featureIndex)
		{
			return Math.exp (values[featureIndex]);
		}

		public double logProbability (int featureIndex)
		{
			return values[featureIndex];
		}

		public void addProbabilities (double[] vector)
		{
			throw new UnsupportedOperationException ("Not implemented.");
		}

		public void addLogProbabilities (double[] vector)
		{
			for (int i = 0; i < values.length; i++)
				vector[i] += values[i];
		}
	}


	// xxx Make this inherit from something like AugmentableDenseFeatureVector
	
	public static abstract class Estimator implements Cloneable
	{
		Alphabet dictionary;
		double counts[];
		int size;														// The number of valid entries in counts[]
		static final int minCapacity = 16;

		protected Estimator (double counts[], int size, Alphabet dictionary)
		{
			this.counts = counts;
			this.size = size;
			this.dictionary = dictionary;
		}

		public Estimator (double counts[], Alphabet dictionary)
		{
			this (counts, dictionary.size(), dictionary);
		}
		
		public Estimator ()
		{
			this (new double[minCapacity], 0, null);
		}

		public Estimator (int size)
		{
			this (new double[size > minCapacity ? size : minCapacity], size, null);
		}

		public Estimator (Alphabet dictionary)
		{
			this(new double[dictionary.size()], dictionary.size(), dictionary);
		}

		public void setAlphabet (Alphabet d)
		{
			this.size = d.size();
			this.counts = new double[size];
			this.dictionary = d;
		}
		
		public int size ()
		{
			return (dictionary == null ? size : dictionary.size());
		}

		private void ensureCapacity (int index)
		{
			//assert (dictionary == null);	// Size is fixed if dictionary present?
			if (index > size)
				size = index;
			if (counts.length <= index) {
				int newLength = ((counts.length < minCapacity)
												 ? minCapacity
												 : counts.length);
				while (newLength <= index)
					newLength *= 2;
				double[] newCounts = new double[newLength];
				System.arraycopy (counts, 0, newCounts, 0, counts.length);
				this.counts = newCounts;
			}
		}

		// xxx Note that this does not reset the "size"!
		public void reset ()
		{
			for (int i = 0; i < counts.length; i++)
				counts[i] = 0;
		}

		// xxx Remove this method?
		private void setCounts (double counts[])
		{
			assert (dictionary == null || counts.length <= size());
			// xxx Copy instead?
			// xxx Set size() to match counts.length?
			this.counts = counts;
		}

		public void increment (int index, double count)
		{
			ensureCapacity (index);
			counts[index] += count;
			if (size < index + 1)
				size = index + 1;
		}

		public void increment (String key, double count)
		{
			increment (dictionary.lookupIndex (key), count);
		}

		// xxx Add "public void increment (Object key, double count)", or is it too dangerous?

		public void increment (FeatureSequence fs, double scale)
		{
			if (fs.getAlphabet() != dictionary)
				throw new IllegalArgumentException ("Vocabularies don't match.");
			for (int fsi = 0; fsi < fs.size(); fsi++)
				increment (fs.getIndexAtPosition(fsi), scale);
		}

		public void increment (FeatureSequence fs)
		{
			increment (fs, 1.0);
		}
		
		public void increment (FeatureVector fv, double scale)
		{
			if (fv.getAlphabet() != dictionary)
				throw new IllegalArgumentException ("Vocabularies don't match.");
			for (int fvi = 0; fvi < fv.numLocations(); fvi++)
				increment (fv.indexAtLocation(fvi), scale);
		}

		public void increment (FeatureVector fv)
		{
			increment (fv, 1.0);
		}

		public double getCount (int index)
		{
			return counts[index];
		}
		
		public Object clone ()
		{
			try {
				return super.clone ();
			} catch (CloneNotSupportedException e) {
				return null;
			}
		}

		public void print () {
			//if (counts != null) throw new IllegalStateException ("Foo");
			System.out.println ("Multinomial.Estimator");
			for (int i = 0; i < size; i++)
				System.out.println ("counts["+i+"] = " + counts[i]);
		}

		public abstract Multinomial estimate ();
		
	}

	public static class MEstimator extends Estimator
	{
		double m;

		public MEstimator (Alphabet dictionary, double m)
		{
			super (dictionary);
			this.m = m;
		}

		
		public MEstimator (int size, double m)
		{
			super(size);
			this.m = m;
		}

		public MEstimator (double m)
		{
			super();
			this.m = m;
		}
		
		public Multinomial estimate ()
		{
			double[] pr = new double[dictionary==null ? size : dictionary.size()];
			double sum = 0;
			for (int i = 0; i < pr.length; i++) {
				pr[i] = counts[i] + m;
				sum += pr[i];
			}
			for (int i = 0; i < pr.length; i++)
				pr[i] /= sum;
			return new Multinomial (pr, dictionary, size, false, false);
		}
		
	}


	public static class MLEstimator extends MEstimator
	{

		public MLEstimator ()
		{
			super (0);
		}
		
		public MLEstimator (int size)
		{
			super (size, 0);
		}

		public MLEstimator (Alphabet dictionary)
		{
			super (dictionary, 0);
		}
		
	}

	
	public static class LaplaceEstimator extends MEstimator
	{

		public LaplaceEstimator ()
		{
			super (1);
		}
		
		public LaplaceEstimator (int size)
		{
			super (size, 1);
		}

		public LaplaceEstimator (Alphabet dictionary)
		{
			super (dictionary, 1);
		}
		
	}
	
	public static class MAPEstimator extends Estimator
	{
		Dirichlet prior;

		public MAPEstimator (Dirichlet d)
		{
			super (d.size());
			prior = d;
		}

		public Multinomial estimate ()
		{
			// xxx unfinished.
			return null;
		}
		
	}

}
