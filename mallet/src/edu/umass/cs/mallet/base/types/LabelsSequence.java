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

public class LabelsSequence implements Sequence
{
	Labels[] seq;
	
	public LabelsSequence (Labels[] seq)
	{
		this.seq = new Labels[seq.length];
		System.arraycopy (seq, 0, this.seq, 0, seq.length);
	}

	public int size () { return seq.length; }

	public Object get (int i) { return seq[i]; }

	public Labels getLabels (int i) { return seq[i]; }
	
}
