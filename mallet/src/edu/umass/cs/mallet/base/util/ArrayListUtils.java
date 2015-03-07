/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.util;

import java.util.ArrayList;

public class ArrayListUtils
{
	static public ArrayList createArrayList (Object[] a)
	{
		ArrayList al = new ArrayList (a.length);
		for (int i = 0; i < a.length; i++)
			al.add (a[i]);
		return al;
	}
	
}
