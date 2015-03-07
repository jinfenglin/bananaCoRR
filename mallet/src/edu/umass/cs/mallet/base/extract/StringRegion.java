/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.extract;

/** A sub-section of a linear string. */

public class StringRegion /* extends Token */ implements Region
{
	String string;
	int start, end;
	
	public StringRegion (String string, int start, int end)
	{
		this.string = string;
		this.start = start;
		this.end = end;
	}
	
	public String getText ()
	{
		return string.substring(start, end);
	}

	public Object getDocument ()
	{
		return string;
	}

	public boolean intersects (Region r)
	{
		if (!(r instanceof StringRegion))
			return false;
		StringRegion sr = (StringRegion)r;
		return (sr.string == this.string && !(sr.end < this.start || sr.start > this.end));
	}

}
