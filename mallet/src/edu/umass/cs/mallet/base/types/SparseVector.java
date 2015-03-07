/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 Sparse, yet its (present) values can be changed.  You can't, however, add
	 values that were (zero and) missing.
	 
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

public class SparseVector extends ConstantVector implements Serializable 
{
	private static Logger logger = MalletLogger.getLogger(SparseVector.class.getName());

	int[] index2location;

	public SparseVector (int[] indices, double[] values, 
											 int capacity, int size,
											 boolean copy,
											 boolean checkIndicesSorted,
											 boolean removeDuplicates)
	{
		super (indices, values, capacity, size, copy, checkIndicesSorted, removeDuplicates);
		assert (indices != null);
	}

	/** Create an empty vector */
	public SparseVector ()
	{
		super (new int[0], new double[0], 0, 0, false, false, false);
	}

	/** Create non-binary vector, possibly dense if "featureIndices" or possibly sparse, if not */
	public SparseVector (int[] featureIndices,
											 double[] values)
	{
		super (featureIndices, values);
	}

	/** Create binary vector */
	public SparseVector (int[] featureIndices)
	{
		super (featureIndices);
	}

	// xxx We need to implement this in FeatureVector subclasses
	public ConstantMatrix cloneMatrix ()
	{
		return new SparseVector (indices, values);
	}

	public ConstantMatrix cloneMatrixZeroed () {
		assert (values != null);
		int[] newIndices = new int[indices.length];
		System.arraycopy (indices, 0, newIndices, 0, indices.length);
		return new SparseVector (newIndices, new double[values.length],
														 values.length, values.length, false, false, false);
	}
	
	// Methods that change values

	private void setIndex2Location ()
	{
		//System.out.println ("SparseVector setIndex2Location indices.length="+indices.length+" maxindex="+indices[indices.length-1]);
		assert (index2location == null);
		assert (indices.length > 0);
		int size = indices[indices.length-1]+1;
		assert (size >= indices.length);
		this.index2location = new int[size];
		Arrays.fill (index2location, -1);
		for (int i = 0; i < indices.length; i++)
			index2location[indices[i]] = i;
	}

	public final void setValue (int index, double value) {
		if (index2location == null)
			setIndex2Location ();
		int location = index < index2location.length ? index2location[index] : -1;
		if (location >= 0)
			values[location] = value;
		else
			throw new IllegalArgumentException ("Trying to set value that isn't present in SparseVector");
	}

	public final void setValueAtLocation (int location, double value)
	{
		values[location] = value;
	}
	
	public final double dotProduct (DenseVector v) {
		double ret = 0;
		if (values == null)
			for (int i = 0; i < indices.length; i++)
				ret += v.value(indices[i]);
		else
			for (int i = 0; i < indices.length; i++)
				ret += values[i] * v.value(indices[i]);
		return ret;
	}
	
	public final double dotProduct (ConstantVector v) {
		if (indices.length == 0)
			return 0;
		if (index2location == null)
			setIndex2Location ();
		double ret = 0;
		if (values == null) {
			for (int i = v.numLocations()-1; i >= 0; i--) {
				int index = v.indexAtLocation(i);
				if (index >= index2location.length)
					break;
				int location = index2location[index];
				if (index2location[location] >= 0)
					ret += v.valueAtLocation (i);
			}
		} else {
			for (int i = v.numLocations()-1; i >= 0; i--) {
				int index = v.indexAtLocation(i);
				if (index >= index2location.length)
					break;
				int location = index2location[index];
				if (location >= 0)
					ret += values[location] * v.valueAtLocation (i);
			}
		}
		return ret;
	}

	public final void plusEquals (ConstantVector v, double factor) {
		if (indices.length == 0)
			return;
		if (index2location == null)
			setIndex2Location ();
		for (int i = v.numLocations()-1; i >= 0; i--) {
			int index = v.indexAtLocation(i);
			if (index >= index2location.length)
				break;
			int location = index2location[index];
			if (location >= 0)
				values[location] += v.valueAtLocation (i) * factor;
		}
	}

	public final void plusEquals (ConstantVector v) {
		if (indices.length == 0)
			return;
		if (index2location == null)
			setIndex2Location ();
		for (int i = v.numLocations()-1; i >= 0; i--) {
			int index = v.indexAtLocation(i);
			if (index >= index2location.length)
				break;
			int location = index2location[index];
			if (location >= 0)
				values[location] += v.valueAtLocation (i);
		}
	}
	
	public final void setAll (double v)
	{
		for (int i = 0; i < values.length; i++)
			values[i] = v;
	}


	
	//Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	static final int NULL_INTEGER = -1;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		if(index2location != null) {
			int size = index2location.length;
			out.writeInt(size);
			for (int i=0; i<size; i++) {
				out.writeInt(index2location[i]);
			}
		}
		else {
			out.writeInt(NULL_INTEGER);
		}
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		int size = in.readInt();
		if (size == NULL_INTEGER) {
			index2location = null;
		}
		else {
			index2location = new int[size];
			for(int i=0; i< size; i++) {
				index2location[i] = in.readInt();
			}
		}
	}

}
