/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.pipe;

import edu.umass.cs.mallet.base.types.*;
import java.io.*;

public class Target2LabelSequence extends Pipe
{

	public Target2LabelSequence ()
	{
		super (null, LabelAlphabet.class);
	}
	
	public Instance pipe (Instance carrier)
	{
		//Object in = carrier.getData();
		Object target = carrier.getTarget();
		if (target instanceof LabelSequence)
			;																	// Nothing to do
		else if (target instanceof TokenSequence) {
			Alphabet v = getTargetAlphabet ();
			TokenSequence ts = (TokenSequence) target;
			int indices[] = new int[ts.size()];
			for (int i = 0; i < ts.size(); i++)
				indices[i] = v.lookupIndex (ts.getToken(i).getText());
			LabelSequence ls = new LabelSequence ((LabelAlphabet)getTargetAlphabet(), indices);
			carrier.setTarget(ls);
		} else {
			throw new IllegalArgumentException ("Unrecognized target type.");
		}
		return carrier;
	}


	// Serialization 
	
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
	}
	
}