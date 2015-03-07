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

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.util.*;
import java.util.*;

public abstract class StringTokenization extends Tokenization
{
	StringRegion[] regions;

	public StringTokenization (String string, CharSequenceLexer lexer)
	{
		super();
		ArrayList rs = new ArrayList();
		lexer.setCharSequence (string);
		while (lexer.hasNext()) {
			this.add (new Token ((String) lexer.next()));
			rs.add (new StringRegion (string, lexer.getStartOffset(), lexer.getEndOffset()));
		}
		regions = (StringRegion[]) rs.toArray();
	}

	public StringTokenization (StringRegion[] stringRegions)
	{
		regions = stringRegions;
		for (int i = 0; i < regions.length; i++)
			this.add (new Token (regions[i].getText()));
	}
	
}
