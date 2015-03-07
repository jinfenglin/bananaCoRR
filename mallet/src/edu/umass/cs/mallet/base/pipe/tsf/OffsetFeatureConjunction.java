/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 Create new features from all possible conjunctions with other
	 (possibly position-offset) features.

   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.pipe.tsf;

import edu.umass.cs.mallet.base.types.TokenSequence;
import edu.umass.cs.mallet.base.util.PropertyList;
import edu.umass.cs.mallet.base.pipe.Pipe;
import edu.umass.cs.mallet.base.types.Token;
import edu.umass.cs.mallet.base.types.Instance;
import java.io.*;

public class OffsetFeatureConjunction extends Pipe implements Serializable
{
	String thisFeatureName;
	String[] featureNames;
	int[] offsets;
	boolean[] negations;
	
	// To include all the old previous singleton features, pass {{0}}
	// For a conjunction at the current time step, pass {{0,0}}
	// For a conjunction of current and previous, pass {{0,-1}}
	// For a conjunction of the current and next two, pass {{0,1,2}}
	public OffsetFeatureConjunction (String thisFeatureName, String[] featureNames, int[] offsets, boolean[] negations)
	{
		this.thisFeatureName = thisFeatureName;
		this.featureNames = featureNames;
		this.offsets = offsets;
		this.negations = negations;
	}

	private static boolean[] trueArray (int length) {
		boolean[] ret = new boolean[length];
		for (int i = 0; i < length; i++)
			ret[i] = true;
		return ret;
	}

	public OffsetFeatureConjunction (String thisFeatureName, String[] featureNames, int[] offsets)
	{
		this (thisFeatureName, featureNames, offsets, trueArray(featureNames.length));
	}

	public Instance pipe (Instance carrier)
	{
		TokenSequence ts = (TokenSequence) carrier.getData();
		int tsSize = ts.size();
		PropertyList[] oldfs = new PropertyList[ts.size()];
		PropertyList[] newfs = new PropertyList[ts.size()];
		boolean passes = true;
		for (int i = 0; i < tsSize; i++) {
			for (int j = 0; j < featureNames.length; j++) {
				int offset = j + offsets[i];
				if (!(offset >= 0 && i + offset < tsSize)) {
					passes = false;
					break;
				}
				if (!(negations[i] ^ ts.getToken(offset).getFeatureValue(featureNames[i]) == 0)) {
					passes = false;
					break;
				}
			}
			if (passes)
				ts.getToken(i).setFeatureValue (thisFeatureName, 1.0);
		}
		return carrier;
	}

	// Serialization 
	
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject (thisFeatureName);
		int size;
		size = (featureNames == null) ? NULL_INTEGER : featureNames.length;
		out.writeInt(size);
		if (size != NULL_INTEGER) {
			for (int i = 0; i <size; i++) {
				out.writeObject (featureNames[i]);
				out.writeInt (offsets[i]);
				out.writeBoolean (negations[i]);
			}
		}
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int size;
		int version = in.readInt ();
		thisFeatureName = (String) in.readObject();
		size = in.readInt();;
		if (size == NULL_INTEGER) {
			featureNames = null;
			offsets = null;
			negations = null;
		}	else {
			featureNames = new String[size];
			offsets = new int[size];
			negations = new boolean[size];
			for (int i = 0; i < size; i++) {
				featureNames[i] = (String) in.readObject();
				offsets[i] = in.readInt();
				negations[i] = in.readBoolean();
			}
		}
	}

}
