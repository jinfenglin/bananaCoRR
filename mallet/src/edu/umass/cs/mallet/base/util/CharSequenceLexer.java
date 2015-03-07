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

import edu.umass.cs.mallet.base.util.Lexer;
import java.io.*;
import java.lang.CharSequence;
import java.util.Iterator;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.io.*;

public class CharSequenceLexer implements Lexer, Serializable
{
	// Some predefined lexing rules
	public static final Pattern LEX_ALPHA = Pattern.compile ("\\p{Alpha}+");
	public static final Pattern LEX_WORDS = Pattern.compile ("\\w+");
	public static final Pattern LEX_NONWHITESPACE_TOGETHER = Pattern.compile ("\\S+");
	public static final Pattern LEX_WORD_CLASSES	=
	Pattern.compile ("\\p{Alpha}+|\\p{Digit}+");
	public static final Pattern LEX_NONWHITESPACE_CLASSES	=
	Pattern.compile ("\\p{Alpha}+|\\p{Digit}+|\\p{Punct}");

	Pattern regex;
	Matcher matcher = null;
	CharSequence input;
	String matchText;
	boolean matchTextFresh;

	public CharSequenceLexer ()
	{
		this (LEX_ALPHA);
	}

	public CharSequenceLexer (Pattern regex)
	{
		this.regex = regex;
		setCharSequence (null);
	}
	
	public CharSequenceLexer (String regex)
	{
		this (Pattern.compile (regex));
	}
	
	public CharSequenceLexer (CharSequence input, Pattern regex)
	{
		this (regex);
		setCharSequence (input);
	}
	
	public CharSequenceLexer (CharSequence input, String regex)
	{
		this (input, Pattern.compile (regex));
	}

	public void setCharSequence (CharSequence input)
	{
		this.input = input;
		this.matchText = null;
		this.matchTextFresh = false;
		if (input != null)
			this.matcher = regex.matcher(input);
	}
	
	public int getStartOffset ()
	{
		if (matchText == null)
			return -1;
		return matcher.start();
	}

	public int getEndOffset ()
	{
		if (matchText == null)
			return -1;
		return matcher.end();
	}

	public String getTokenString ()
	{
		return matchText;
	}

	
	// Iterator interface methods

	private void updateMatchText ()
	{
		if (matcher != null && matcher.find()) {
			matchText = matcher.group();
			if (matchText.length() == 0) {
				// xxx Why would this happen?
				// It is happening to me when I use the regex ".*" in an attempt to make
				// Token's out of entire lines of text. -akm.
				updateMatchText();
				//System.err.println ("Match text is empty!");
			}
		//matchText = input.subSequence (matcher.start(), matcher.end()).toString ();
		} else
			matchText = null;
		matchTextFresh = true;
	}

	public boolean hasNext ()
	{
		if (! matchTextFresh)
			updateMatchText ();
		return (matchText != null);
	}

	public Object next ()
	{
		if (! matchTextFresh)
			updateMatchText ();
		matchTextFresh = false;
		return matchText;
	}

	public void remove ()
	{
		throw new UnsupportedOperationException ();
	}

	// Serialization 
	
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject(regex);
		//out.writeObject(matcher);
		//out.writeObject(input);
		//out.writeObject(matchText);
		//out.writeBoolean(matchTextFresh);
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		regex = (Pattern) in.readObject();
		matcher = null;
		input = null;
		matchText = null;
		matchTextFresh = false;
	}

	public static void main (String[] args)
	{
		try {
			BufferedReader in
				= new BufferedReader(new FileReader(args[0]));
			for (String line = in.readLine(); line != null; line = in.readLine()) {
				CharSequenceLexer csl =
					new CharSequenceLexer (line, LEX_NONWHITESPACE_CLASSES );
				while (csl.hasNext())
					System.out.println (csl.next());
			}
		} catch (Exception e) {
			System.out.println (e.toString());
		}
	}
	
}