/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

// Rename to Segment, (then also Segmentation, SegmentSequence, SegmentList)
// Alternatively, think about names: Annotation, AnnotationList, 

package edu.umass.cs.mallet.base.extract;

/** A sub-section of a document, either linear or two-dimensional */

public interface Region
{

	public abstract String getText ();

	public abstract Object getDocument ();

	public abstract boolean intersects (Region r);

}
