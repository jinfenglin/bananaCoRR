/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 A collection of labels, either for a multi-label problem (all
	 labels are part of the same label dictionary), or a factorized
	 labeling, (each label is part of a different dictionary).
	 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.types;

import edu.umass.cs.mallet.base.types.Label;

/** Usually some distribution over possible labels for an instance. */

public class Labels
{
	Label[] labels;
	
	public Labels (Label[] labels)
	{
		this.labels = new Label[labels.length];
		System.arraycopy (labels, 0, this.labels, 0, labels.length);
	}

	// Number of factors
	public int size () { return labels.length; }

	public Label get (int i) { return labels[i]; }
	
}
