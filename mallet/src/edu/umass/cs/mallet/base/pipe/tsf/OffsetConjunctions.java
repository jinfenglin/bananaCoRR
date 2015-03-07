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
import java.util.regex.*;

public class OffsetConjunctions extends Pipe implements Serializable
{
	int[][] conjunctions;
	boolean includeOriginalSingletons;
//	boolean includeBeginEndBoundaries;
	Pattern featureRegex;

	static final int maxWindowSize = 20;
	static final PropertyList[] startfs = new PropertyList[maxWindowSize];
	static final PropertyList[] endfs = new PropertyList[maxWindowSize];

	static {
		initStartEndFs ();
	}

	private static void initStartEndFs ()
	{
		for (int i = 0; i < maxWindowSize; i++) {
			startfs[i] = PropertyList.add ("<START"+i+">", 1.0, null);
			endfs[i] = PropertyList.add ("<END"+i+">", 1.0, null);
		}
	}
	
	// To include all the old previous singleton features, pass {{0}}
	// For a conjunction at the current time step, pass {{0,0}}
	// For a conjunction of current and previous, pass {{0,-1}}
	// For a conjunction of the current and next two, pass {{0,1,2}}
	public OffsetConjunctions (boolean includeOriginalSingletons, Pattern featureRegex, int[][] conjunctions)
	{
		this.conjunctions = conjunctions;
		this.featureRegex = featureRegex;
		this.includeOriginalSingletons = includeOriginalSingletons;
	}

	public OffsetConjunctions (boolean includeOriginalSingletons, int[][] conjunctions)
	{
		this (true, null, conjunctions);
	}
	
	public OffsetConjunctions (int[][] conjunctions)
	{
		this (true, conjunctions);
	}
	
	public Instance pipe (Instance carrier)
	{
		TokenSequence ts = (TokenSequence) carrier.getData();
		int tsSize = ts.size();
		PropertyList[] oldfs = null;
		PropertyList[] newfs = null;
		try {
			oldfs = new PropertyList[ts.size()];
		}
		catch (Exception e) {
			System.err.println("Exception allocating oldfs: " + e);
		}
		try {
			newfs = new PropertyList[ts.size()];
		}
		catch (Exception e) {
			System.err.println("Exception allocating newfs: " + e);
		}
		
		for (int i = 0; i < tsSize; i++)
			oldfs[i] = ts.getToken(i).getFeatures ();
		if (includeOriginalSingletons)
			for (int i = 0; i < tsSize; i++)
				newfs[i] = ts.getToken(i).getFeatures ();

		for (int i = 0; i < tsSize; i++) {
			//System.out.println ("OffsetConjunctions: ts index="+i+", conjunction =");
			conjunctionList: for (int j = 0; j < conjunctions.length; j++) {
				// Make sure that the offsets in the conjunction are all available at this position
				for (int k = 0; k < conjunctions[j].length; k++) {
					//if (conjunctions[j][k] + i < 0 || conjunctions[j][k] + i > tsSize-1	|| oldfs[i+conjunctions[j][k]] == null)
					//continue conjunctionList;
					//System.out.print (" "+conjunctions[j][k]);
				}
				//System.out.print ("\n");

				// Add the features for this conjunction
				if (conjunctions[j].length == 1) {
					int offset = conjunctions[j][0];
					if (offset == 0 && includeOriginalSingletons)
						throw new IllegalArgumentException ("Original singletons already there.");
					PropertyList.Iterator iter;
					if (i+offset < 0)
						iter = startfs[-(i+offset)-1].iterator();
					else if (offset+i > tsSize-1)
						iter = endfs[i+offset-tsSize].iterator();
					else if (oldfs[offset+i] == null)
						continue conjunctionList;
					else
						iter = oldfs[i+offset].iterator();
					while (iter.hasNext()) {
						iter.next();
						String k = iter.getKey();
						if (featureRegex != null && !featureRegex.matcher(k).matches())
							continue;
						String key = k + (offset==0 ? "" : "@"+offset);
						try {
							newfs[i] = PropertyList.add (key, iter.getNumericValue(), newfs[i]);
						}
						catch (Exception e) {
							System.err.println("Adding to property list length 1: " + e);
						}
					}

				} else if (conjunctions[j].length == 2) {
					//System.out.println ("token="+ts.getToken(i).getText()+" conjunctionIndex="+j);
					int offset0 = conjunctions[j][0];
					int offset1 = conjunctions[j][1];
					PropertyList.Iterator iter0;
					if (i+offset0 < 0)
						iter0 = startfs[-(i+offset0)-1].iterator();
					else if (offset0+i > tsSize-1)
						iter0 = endfs[i+offset0-tsSize].iterator();
					else if (oldfs[offset0+i] == null)
						continue conjunctionList;
					else
						iter0 = oldfs[i+offset0].iterator();
					int iter0i = -1;
					while (iter0.hasNext()) {
						iter0i++;
						iter0.next();
						String k0 = iter0.getKey();
						if (featureRegex != null && !featureRegex.matcher(k0).matches())
							continue;
						PropertyList.Iterator iter1;
						if (i+offset1 < 0)
							iter1 = startfs[-(i+offset1)-1].iterator();
						else if (offset1+i > tsSize-1)
							iter1 = endfs[i+offset1-tsSize].iterator();
						else if (oldfs[offset1+i] == null)
							continue conjunctionList;
						else
							iter1 = oldfs[i+offset1].iterator();
						int iter1i = -1;
						while (iter1.hasNext()) {
							iter1i++;
							iter1.next();
							String k1 = iter1.getKey();
							if (featureRegex != null && !featureRegex.matcher(k1).matches())
								continue;
							// Avoid redundant doubling of feature space; include only upper triangle
							//System.out.println ("off0="+offset0+" off1="+offset1+" iter0i="+iter0i+" iter1i="+iter1i);
							if (offset0 == offset1 && iter1i <= iter0i) continue; 
							//System.out.println (">off0="+offset0+" off1="+offset1+" iter0i="+iter0i+" iter1i="+iter1i);
							String key = iter0.getKey() + (offset0==0 ? "" : "@"+offset0)
													 +"_&_"+iter1.getKey() + (offset1==0 ? "" : "@"+offset1);
							try {
								newfs[i] = PropertyList.add (key, iter0.getNumericValue() * iter1.getNumericValue(), newfs[i]);
							}
							catch (Exception e) {
								System.err.println("Adding to property list length 2: " + e);
							}
							
						}
					}

				} else if (conjunctions[j].length == 3) {
					int offset0 = conjunctions[j][0];
					int offset1 = conjunctions[j][1];
					int offset2 = conjunctions[j][2];
					PropertyList.Iterator iter0;
					if (i+offset0 < 0)
						iter0 = startfs[-(i+offset0)-1].iterator();
					else if (offset0+i > tsSize-1)
						iter0 = endfs[i+offset0-tsSize].iterator();
					else if (oldfs[offset0+i] == null)
						continue conjunctionList;
					else
						iter0 = oldfs[i+offset0].iterator();
					int iter0i = -1;
					while (iter0.hasNext()) {
						iter0i++;
						iter0.next();
						String k0 = iter0.getKey();
						if (featureRegex != null && !featureRegex.matcher(k0).matches())
							continue;
						PropertyList.Iterator iter1;
						if (i+offset1 < 0)
							iter1 = startfs[-(i+offset1)-1].iterator();
						else if (offset1+i > tsSize-1)
							iter1 = endfs[i+offset1-tsSize].iterator();
						else if (oldfs[offset1+i] == null)
							continue conjunctionList;
						else
							iter1 = oldfs[i+offset1].iterator();
						int iter1i = -1;
						while (iter1.hasNext()) {
							iter1i++;
							iter1.next();
							String k1 = iter1.getKey();
							if (featureRegex != null && !featureRegex.matcher(k1).matches())
								continue;
							// Avoid redundant doubling of feature space; include only upper triangle
							if (offset0 == offset1 && iter1i <= iter0i) continue; 
							PropertyList.Iterator iter2;
							if (i+offset2 < 0)
								iter2 = startfs[-(i+offset2)-1].iterator();
							else if (offset2+i > tsSize-1)
								iter2 = endfs[i+offset2-tsSize].iterator();
							else if (oldfs[offset2+i] == null)
								continue conjunctionList;
							else
								iter2 = oldfs[i+offset2].iterator();
							int iter2i = -1;
							while (iter2.hasNext()) {
								iter2i++;
								iter2.next();
								String k2 = iter2.getKey();
								if (featureRegex != null && !featureRegex.matcher(k2).matches())
									continue;
								// Avoid redundant doubling of feature space; include only upper triangle
								if (offset1 == offset2 && iter2i <= iter1i) continue; 
								String key = iter0.getKey() + (offset0==0 ? "" : "@"+offset0)
														 +"_&_"+iter1.getKey() + (offset1==0 ? "" : "@"+offset1)
														 +"_&_"+iter2.getKey() + (offset2==0 ? "" : "@"+offset2);
								try {
									newfs[i] = PropertyList.add (key, iter0.getNumericValue() * iter1.getNumericValue()
																						 * iter2.getNumericValue(), newfs[i]);
								}
								catch (Exception e) {
									System.err.println("Adding to property list length 3: " + e);
								}
					
							}
						}
					}

				} else if (conjunctions[j].length == 4) {
					int offset0 = conjunctions[j][0];
					int offset1 = conjunctions[j][1];
					int offset2 = conjunctions[j][2];
					int offset3 = conjunctions[j][3];
					PropertyList.Iterator iter0;
					if (i+offset0 < 0)
						iter0 = startfs[-(i+offset0)-1].iterator();
					else if (offset0+i > tsSize-1)
						iter0 = endfs[i+offset0-tsSize].iterator();
					else if (oldfs[offset0+i] == null)
						continue conjunctionList;
					else
						iter0 = oldfs[i+offset0].iterator();
					int iter0i = -1;
					while (iter0.hasNext()) {
						iter0i++;
						iter0.next();
						PropertyList.Iterator iter1;
						if (i+offset1 < 0)
							iter1 = startfs[-(i+offset1)-1].iterator();
						else if (offset1+i > tsSize-1)
							iter1 = endfs[i+offset1-tsSize].iterator();
						else if (oldfs[offset1+i] == null)
							continue conjunctionList;
						else
							iter1 = oldfs[i+offset1].iterator();
						int iter1i = -1;
						while (iter1.hasNext()) {
							iter1i++;
							iter1.next();
							// Avoid redundant doubling of feature space; include only upper triangle
							if (offset0 == offset1 && iter1i <= iter0i) continue; 
							PropertyList.Iterator iter2;
							if (i+offset2 < 0)
								iter2 = startfs[-(i+offset2)-1].iterator();
							else if (offset2+i > tsSize-1)
								iter2 = endfs[i+offset2-tsSize].iterator();
							else if (oldfs[offset2+i] == null)
								continue conjunctionList;
							else
								iter2 = oldfs[i+offset2].iterator();
							int iter2i = -1;
							while (iter2.hasNext()) {
								iter2i++;
								iter2.next();
								// Avoid redundant doubling of feature space; include only upper triangle
								if (offset1 == offset2 && iter2i <= iter1i) continue;
								PropertyList.Iterator iter3;
								if (i+offset3 < 0)
									iter3 = startfs[-(i+offset3)-1].iterator();
								else if (offset3+i > tsSize-1)
									iter3 = endfs[i+offset3-tsSize].iterator();
								else if (oldfs[offset3+i] == null)
									continue conjunctionList;
								else
									iter3 = oldfs[i+offset3].iterator();
								int iter3i = -1;
								while (iter3.hasNext()) {
									iter3i++;
									iter3.next();
									// Avoid redundant doubling of feature space; include only upper triangle
									if (offset2 == offset3 && iter3i <= iter2i) continue;
									String key = iter0.getKey() + (offset0==0 ? "" : "@"+offset0)
															 +"_&_"+iter1.getKey() + (offset1==0 ? "" : "@"+offset1)
															 +"_&_"+iter2.getKey() + (offset2==0 ? "" : "@"+offset2)
															 +"_&_"+iter3.getKey() + (offset3==0 ? "" : "@"+offset3);
									newfs[i] = PropertyList.add (key, iter0.getNumericValue() * iter1.getNumericValue()
																							 * iter2.getNumericValue() * iter3.getNumericValue(), newfs[i]);
								}
							}
						}
					}
					
				} else {
					throw new UnsupportedOperationException ("Conjunctions of length 4 or more not yet implemented.");
				}
			}
		}

		// Put the new PropertyLists in place
		for (int i = 0; i < ts.size(); i++)
			ts.getToken(i).setFeatures (newfs[i]);
		//System.err.println("Compeleted pipe");
		return carrier;
	}

	// Serialization 
	
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		int size1, size2;
		size1 = (conjunctions == null) ? NULL_INTEGER : conjunctions.length;
		out.writeInt(size1);
		if (size1 != NULL_INTEGER) {
			for (int i = 0; i <size1; i++) {
				size2 = (conjunctions[i] == null) ? NULL_INTEGER: conjunctions[i].length;
				out.writeInt(size2);
				if (size2 != NULL_INTEGER) {
					for (int j = 0; j <size2; j++) {
						out.writeInt(conjunctions[i][j]);
					}
				}
			}
		}
		out.writeBoolean(includeOriginalSingletons);
		
		out.writeObject(featureRegex); //add by fuchun
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int size1, size2;
		int version = in.readInt ();
		size1 = in.readInt();
		// Deserialization doesn't call the unnamed class initializer, so do it here
		if (startfs[0] == null)
			initStartEndFs ();
		if (size1 == NULL_INTEGER) {
			conjunctions = null;
		}
		else {
			conjunctions = new int[size1][];
			for (int i = 0; i < size1; i++) {
				size2 = in.readInt();
				if (size2 == NULL_INTEGER) {
					conjunctions[i] = null;
				}
				else {
					conjunctions[i] = new int[size2];
					for (int j = 0; j < size2; j++) {
						conjunctions[i][j] = in.readInt();
					}
				}
			}
		}
		includeOriginalSingletons = in.readBoolean();
		featureRegex = (Pattern) in. readObject();//add by fuchun
	
	}
}
