/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 Run a regular expression over the text of each token; replace the
	 text with the substring matching one regex group; create a target
	 TokenSequence from the text matching another regex group.

	 <p>For example, if you have a data file containing one line per token,
	 and the label also appears on that line, you can first get a
	 TokenSequence in which the text of each line is the Token.getText()
	 of each token, then run this pipe, and separate the target
	 information from the data information.  For example to process the
	 following,

	 <pre>
	 BACKGROUND Then
	 PERSON Mr.
	 PERSON Smith
	 BACKGROUND said
	 ...
	 </pre>

	 use <code>new TokenSequenceMatchDataAndTarget (Pattern.compile ("([A-Z]+) (.*)"), 2, 1)</code>.
	 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.pipe;

import edu.umass.cs.mallet.base.types.TokenSequence;
import edu.umass.cs.mallet.base.types.Token;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.util.MalletLogger;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.logging.*;
import java.io.*;

public class TokenSequenceMatchDataAndTarget extends Pipe implements Serializable
{
	private static Logger logger = MalletLogger.getLogger(TokenSequenceMatchDataAndTarget.class.getName());

	Pattern regex;
	int dataGroup;
	int targetGroup;
	
	public TokenSequenceMatchDataAndTarget (Pattern regex, int dataGroup, int targetGroup)
	{
		this.regex = regex;
		this.dataGroup = dataGroup;
		this.targetGroup = targetGroup;
	}

	public TokenSequenceMatchDataAndTarget (String regex, int dataGroup, int targetGroup)
	{
		this (Pattern.compile (regex), dataGroup, targetGroup);
	}

	public Instance pipe (Instance carrier)
	{
		TokenSequence ts = (TokenSequence) carrier.getData();
		TokenSequence targetTokenSeq = new TokenSequence (ts.size());
		for (int i = 0; i < ts.size(); i++) {
			Token t = ts.getToken(i);
			Matcher matcher = regex.matcher (t.getText());
			if (matcher.matches()) {
				targetTokenSeq.add (matcher.group(targetGroup));
				t.setText (matcher.group (dataGroup));
			} else {
				logger.warning ("Skipping token: No match of "+regex.pattern()
												+" at token #"+i+" with text "+t.getText());
			}
		}
		carrier.setTarget(targetTokenSeq);
		carrier.setData(ts);
		return carrier;
	}

	// Serialization 
	
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject(regex);
		out.writeInt(dataGroup);
		out.writeInt(targetGroup);
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		regex = (Pattern) in.readObject();
		dataGroup = in.readInt();
		targetGroup = in.readInt();
	}

}
