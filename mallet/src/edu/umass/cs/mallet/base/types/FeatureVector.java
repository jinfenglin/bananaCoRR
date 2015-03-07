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

import edu.umass.cs.mallet.base.types.Vector;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.FeatureSequence;
import edu.umass.cs.mallet.base.util.MalletLogger;
import edu.umass.cs.mallet.base.util.PropertyList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Arrays;
import java.util.logging.*;
import java.io.*;

public class FeatureVector extends ConstantVector implements Serializable 
{
	private static Logger logger = MalletLogger.getLogger(FeatureVector.class.getName());

	Alphabet dictionary;
	
	protected FeatureVector (Alphabet dict,
													 int[] indices, double[] values, 
													 int capacity, int size,
													 boolean copy,
													 boolean checkIndicesSorted,
													 boolean removeDuplicates)
	{
		super (indices, values, capacity, size, copy, checkIndicesSorted, removeDuplicates);
		this.dictionary = dict;
	}

	/** Create a dense vector */
	public FeatureVector (Alphabet dict, double[] values)
	{
		super (values);
		this.dictionary = dict;
	}
	
	/** Create non-binary vector, possibly dense if "featureIndices" or possibly sparse, if not */
	public FeatureVector (Alphabet dict,
												int[] featureIndices,
												double[] values)
	{
		super (featureIndices, values);
		this.dictionary = dict;
	}

	/** Create binary vector */
	public FeatureVector (Alphabet dict,
												int[] featureIndices)
	{
		super (featureIndices);
		this.dictionary = dict;
	}

	public static int[] getObjectIndices(Object[] entries, Alphabet dict, boolean addIfNotPresent)
	{
		int[] feats = new int[entries.length];
		for (int i = 0; i < entries.length; i++) {
			feats[i] = dict.lookupIndex (entries[i], addIfNotPresent);
			if (feats[i] == -1)
				throw new IllegalArgumentException ("Object is not in dictionary.");
		}
		return feats;
	}
		
	public FeatureVector (Alphabet dict, Object[] keys, double[] values)
	{
		this (dict, getObjectIndices(keys, dict, true), values);
	}

	private static int[] sortedFeatureIndexSequence (FeatureSequence fs)
	{
		int[] feats = fs.toFeatureIndexSequence ();
		java.util.Arrays.sort (feats);
		return feats;
	}

	public FeatureVector (FeatureSequence fs, boolean binary)
	{
		super (fs.toSortedFeatureIndexSequence(), false, false, true, binary);
		this.dictionary = (Alphabet) fs.getAlphabet();
	}

	public FeatureVector (FeatureSequence fs)
	{
		this (fs, false);
	}
	
	public FeatureVector (Alphabet dict, PropertyList pl, boolean binary,
												boolean growAlphabet)
	{
		super (dict, pl, binary, growAlphabet);
		this.dictionary = dict;
	}

	public FeatureVector (Alphabet dict, PropertyList pl, boolean binary) {
		this (dict, pl, binary, true);
	}

	private static int[] indicesWithConjunctions (FeatureVector fv, Alphabet newVocab, int[] conjunctions)
	{
		assert (fv.values == null);					// Only works on binary feature vectors
		assert (! (fv instanceof AugmentableFeatureVector));
		Alphabet v = fv.getAlphabet();
		// newVocab should be an augmented copy of v
		assert (v.size() <= newVocab.size())
												: "fv.vocab.size="+v.size()+" newVocab.size="+newVocab.size();
		int[] newIndices = new int[fv.indices.length * conjunctions.length];
		java.util.Arrays.sort (conjunctions);
		System.arraycopy (fv.indices, 0, newIndices, 0, fv.indices.length);
		int size = fv.indices.length;
		int ci = 0;
		for (int i = 0; i < fv.indices.length; i++) {
			if (ci < conjunctions.length && conjunctions[ci] < fv.indices[i])
				ci++;
			if (conjunctions[ci] == fv.indices[i]) {
				for (int j = 0; j < fv.indices.length; j++) {
					if (conjunctions[ci] != fv.indices[j]) {
						int index = newVocab.lookupIndex (FeatureConjunction.getName (v, conjunctions[ci], fv.indices[j]));
						if (index == newVocab.size()-1 && index % 3 == 0)
							logger.info ("New feature "+ newVocab.lookupObject(index));
						newIndices[size++] = index;
					}
				}
			}
		}
		int[] ret = new int[size];
		System.arraycopy (newIndices, 0, ret, 0, size);
		return ret;
	}

	private static int[] indicesWithConjunctions (FeatureVector fv, Alphabet newVocab,
																								FeatureSelection fsNarrow,
																								FeatureSelection fsWide)
	{
		assert (fv.values == null);					// Only works on binary feature vectors
		////assert (! (fv instanceof AugmentableFeatureVector));
		Alphabet v = fv.getAlphabet();
		// newVocab should be an augmented copy of v
		assert (v.size() <= newVocab.size())
												: "fv.vocab.size="+v.size()+" newVocab.size="+newVocab.size();
		int length;
		if (fv instanceof AugmentableFeatureVector) {
			length = ((AugmentableFeatureVector)fv).size;
			((AugmentableFeatureVector)fv).sortIndices();
		} else {
			length = fv.indices.length;
		}
		int[] newIndices = new int[length * length];
		System.arraycopy (fv.indices, 0, newIndices, 0, length);
		int size = length;
		int ci = 0;
		for (int i = 0; i < length; i++) {
			if (fsNarrow != null && !fsNarrow.contains (fv.indices[i]))
				continue;
			for (int j = 0; j < length; j++) {
				if ((fsWide == null || fsWide.contains (fv.indices[j]))
						&& fv.indices[i] != fv.indices[j]
						//&& !FeatureConjunction.featuresOverlap (v, fv.indices[i], fv.indices[j]))
					)
				{
					int index = newVocab.lookupIndex (FeatureConjunction.getName (v, fv.indices[i], fv.indices[j]));
					//if (index == newVocab.size()-1 && index % 50 == 0)
					//System.out.println ("FeatureVector: Conjunction feature "+ newVocab.lookupObject(index));
					newIndices[size++] = index;
				}
			}
		}

		// Sort and remove duplicates
		Arrays.sort (newIndices, 0, size);
		for (int i = 1; i < size; i++) {
			if (newIndices[i-1] == newIndices[i]) {
				for (int j = i+1; j < size; j++)
					newIndices[j-1] = newIndices[j];
				size--;
			}
		}
		
		int[] ret = new int[size];
		System.arraycopy (newIndices, 0, ret, 0, size);
		return ret;
	}
	
	/** New feature vector containing all the features of "fv", plus new
			features created by making conjunctions between the features in
			"conjunctions" and all the other features. */
	public FeatureVector (FeatureVector fv, Alphabet newVocab, int[] conjunctions)
	{
		this (newVocab, indicesWithConjunctions (fv, newVocab, conjunctions));
	}

	public FeatureVector (FeatureVector fv, Alphabet newVocab,
												FeatureSelection fsNarrow, FeatureSelection fsWide)
	{
		this (newVocab, indicesWithConjunctions (fv, newVocab, fsNarrow, fsWide));
	}

	// xxx We need to implement this in FeatureVector subclasses
	public ConstantMatrix cloneMatrix ()
	{
		return new FeatureVector ((Alphabet)dictionary, indices, values);
	}

	public ConstantMatrix cloneMatrixZeroed () {
		assert (values != null);
		if (indices == null)
			return new FeatureVector (dictionary, new double[values.length]);
		else {
			int[] newIndices = new int[indices.length];
			System.arraycopy (indices, 0, newIndices, 0, indices.length);
			return new FeatureVector (dictionary, newIndices, new double[values.length],
																values.length, values.length, false, false, false);
		}
	}
	
	public String toString ()
	{
		return toString (false);
	}
	
	public String toString (boolean onOneLine)
	{
		//Thread.currentThread().dumpStack();
		StringBuffer sb = new StringBuffer ();
		//System.out.println ("FeatureVector toString dictionary="+dictionary);
		if (values == null) {
			//System.out.println ("FeatureVector toString values==null");
			int indicesLength = numLocations();
			for (int i = 0; i < indicesLength; i++) {
				//System.out.println ("FeatureVector toString i="+i);
				if (dictionary == null)
					sb.append ("["+i+"]");
				else {
					//System.out.println ("FeatureVector toString: i="+i+" index="+indices[i]);
					sb.append (dictionary.lookupObject(indices[i]).toString());
					//sb.append ("("+indices[i]+")");
				}
				//sb.append ("= 1.0 (forced binary)");
				if (!onOneLine)
					sb.append ('\n');
				else
					sb.append (' ');
			}
		} else {
			//System.out.println ("FeatureVector toString values!=null");
			int valuesLength = numLocations();
			for (int i = 0; i < valuesLength; i++) {
				if (dictionary == null)
					sb.append ("["+i+"]");
				else
					sb.append (dictionary.lookupObject(indices == null ? i : indices[i]).toString());
				sb.append ("=");
				sb.append (values[i]);
				if (!onOneLine)
					sb.append ("\n");
				else
					sb.append (' ');
			}
		}
		return sb.toString();
	}

	public Alphabet getAlphabet ()
	{
		return dictionary;
	}

	public int location (Object entry)
	{
		if (dictionary == null)
			throw new IllegalStateException ("This FeatureVector has no dictionary.");
		int i = dictionary.lookupIndex (entry, false);
		if (i < 0)
			return -1;
		else
			return location (i);
	}

	public boolean contains (Object entry)
	{
		int loc = location(entry);
		return (loc >= 0 && valueAtLocation(loc) != 0);
	}

	public double value (Object o)
	{
		int loc = location (o);
		if (loc >= 0)
			return valueAtLocation (loc);
		else
			throw new IllegalArgumentException ("Object is not a key in the dictionary.");
	}
	
	//Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject (dictionary);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		dictionary = (Alphabet) in.readObject();
	}

}
