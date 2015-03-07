/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 The abstract superclass of all Pipe steps, which transform one data type to another,
	 often used for feature extraction.

   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.pipe;

import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Alphabet;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.util.MalletLogger;
import edu.umass.cs.mallet.base.util.PropertyList;
import java.lang.reflect.Method;
import java.net.URI;
import java.lang.reflect.InvocationTargetException;
import java.util.logging.*;
import java.io.*;

public abstract class Pipe implements Serializable
{
	private static Logger logger = MalletLogger.getLogger(Pipe.class.getName());

	Pipe parent = null;
	Alphabet dataDict = null;
	Alphabet targetDict = null;

	// If non-null, then get*Alphabet methods are guaranteed to return non-null,
	// and a member of this class
	Class dataDictClass = null;						
	Class targetDictClass = null;
	
	boolean dataAlphabetResolved = false;
	boolean targetAlphabetResolved = false;
	boolean targetProcessing = true;

	
	/** Pass non-null if you want the given dictionary created as an instance of the class */
	public Pipe (Class dataDictClass,
							 Class targetDictClass)
	{
		assert (dataDictClass == null || Alphabet.class.isAssignableFrom (dataDictClass));
		assert (targetDictClass == null || Alphabet.class.isAssignableFrom (targetDictClass));
		this.dataDictClass = dataDictClass;
		this.targetDictClass = targetDictClass;
	}

	public Pipe ()
	{
		this ((Class)null, (Class)null);
	}

	/** Note that, since the default values of the dataDictClass and targetDictClass are null,
			that if you specify null for one of the arguments here, this pipe step will not ever create
			any corresponding dictionary for the argument. */
	public Pipe (Alphabet dataDict,
							 Alphabet targetDict)
	{
		this.dataDict = dataDict;
		this.targetDict = targetDict;
		// Is doesn't matter what the dataDictClass and targetDictClass
		// because they will never get used, now that we have already
		// allocated dictionaries in place.
	}
	
	public abstract Instance pipe (Instance carrier);


	public Instance pipe (Object data, Object target, Object name, Object source,
												Instance parent, PropertyList properties)
	{
		return pipe (new Instance (data, target, name, source));
	}

	/** If argument is false, don't expect to find input material for the target.
			By default, this is true. */
	public void setTargetProcessing (boolean lookForAndProcessTarget)
	{
		targetProcessing = lookForAndProcessTarget;
	}

	/** Return true iff this pipe expects and processes information in
			the <tt>target</tt> slot. */
	public boolean isTargetProcessing ()
	{
		return targetProcessing;
	}

	// Note: This must be called *before* this Pipe has been added to
	// the parent's collection of pipes, otherwise in
	// DictionariedPipe.setParent() we will simply get back this Pipe's
	// Alphabet information.
	public void setParent (Pipe p)
	{
		// Force the user to explicitly set to null before making changes.
		if (p != null) {
			//logger.info ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n"
			//+ "setParent " + p.getClass().getName());
			if (parent != null)
				throw new IllegalStateException ("Parent already set.");
			// xxx ?? assert (!dataAlphabetResolved);
		} else {
			logger.info ("Setting parent to null.");
			Thread.currentThread().dumpStack();
		}
		parent = p;
	}

	public Pipe getParent () { return parent; }

	public Pipe getParentRoot () {
		if (parent == null)
			return this;
		Pipe p = parent;
		while (p.parent != null)
			p = p.parent;
		return p;
	}

	protected Alphabet resolveDataAlphabet ()
	{
		//Thread.dumpStack ();
		//assert (parent != null);
		// xxx This is the problem assert (parent.dataDict != null);
		if (dataAlphabetResolved)
			throw new IllegalStateException ("Data Alphabet already resolved.");
		Alphabet pfd = parent == null ? null : parent.dataDict;
		if (pfd == null) {
			if (dataDict == null && dataDictClass != null) {
				try {
					dataDict = (Alphabet) dataDictClass.newInstance();
				} catch (Exception e) {
					throw new IllegalStateException ("Cannot create new data dictionary of class "+dataDictClass.getName());
				}
				logger.fine ("Creating data  Alphabet.");
			}
		} else {
			if (dataDict == null) {
				// This relies on the fact that these methods are called in order by
				// parents, and even if the parent has a different "dataDict" in its final
				// output, right now "parent.dataDict" will be whatever dictionary is coming
				// to this pipe at (that potentially intermediate) stage of the pipeline.
				dataDict = pfd;
				logger.fine ("Assigning data Alphabet from above.");
			}	else if (!dataDict.equals (pfd))
				throw new IllegalStateException ("Parent and local data Alphabet do not match.");
			else
				logger.fine ("Data Alphabet already matches.");
		}
		//assert (dataDict != null);
		dataAlphabetResolved = true;
		return dataDict;
	}
	
	protected Alphabet resolveTargetAlphabet ()
	{
		if (targetAlphabetResolved)
			throw new IllegalStateException ("Target Alphabet already resolved.");
		Alphabet pld = parent == null ? null : parent.targetDict;
		if (pld == null) {
			if (targetDict == null && targetDictClass != null)
				try {
					targetDict = (Alphabet) targetDictClass.newInstance();
				} catch (Exception e) {
					throw new IllegalStateException ("Cannot create new target dictionary of class "+targetDictClass.getName());
				}
		} else {
			if (targetDict == null)
				// This relies on the fact that these methods are called in order by
				// parents, and even if the parent has a different "targetDict" in its final
				// output, right now "parent.targetDict" will be whatever dictionary is coming
				// to this pipe at (that potentially intermediate) stage of the pipeline.
				targetDict = pld;
			else if (!targetDict.equals (pld))
				throw new IllegalStateException ("Parent and local target Alphabet do not match.");
		}
		//assert (targetDict != null);
		targetAlphabetResolved = true;
		return targetDict;
	}

	// If this Pipe produces objects that use a Alphabet, this
	// method returns that dictionary.  Even if this particular Pipe
	// doesn't use a Alphabet it may return non-null if
	// objects passing through it use a dictionary.

	// This method should not be called until the dictionary is really
	// needed, because it may set off a chain of events that "resolve"
	// the dictionaries of an entire pipeline, and generally this
	// resolution should not take place until the pipeline is completely
	// in place, and pipe() is being called.  
  // xxx Perhaps desire to wait until pipe() is being called is unrealistic
	// and unnecessary.

	public Alphabet getDataAlphabet ()
	{
		//Thread.dumpStack();
		if (!dataAlphabetResolved)
			getParentRoot().resolveDataAlphabet();
		assert (dataAlphabetResolved);
		return dataDict;
	}

	public Alphabet getTargetAlphabet ()
	{
		if (!targetAlphabetResolved)
			getParentRoot().resolveTargetAlphabet();
		assert (targetAlphabetResolved);
		return targetDict;
	}

	public void setDataAlphabet (Alphabet dDict)
	{
		if (dataDict != null)
			throw new IllegalStateException
				("Can't set this Pipe's Data  Alphabet; it already has one.");
		dataDict = dDict;
	}

	public void setTargetAlphabet (Alphabet tDict)
	{
		if (targetDict != null)
			throw new IllegalStateException
				("Can't set this Pipe's Target Alphabet; it already has one.");
		targetDict = tDict;
	}

	
	
	// Serialization 
		
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject(parent);
		out.writeObject(dataDict);
		out.writeObject(targetDict);
		out.writeBoolean(dataAlphabetResolved);
		out.writeBoolean(targetAlphabetResolved);
		out.writeObject(dataDictClass);
		out.writeObject(targetDictClass);

	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		parent = (Pipe) in.readObject();
		dataDict = (Alphabet) in.readObject();
		targetDict = (Alphabet) in.readObject();
		dataAlphabetResolved = in.readBoolean();
		targetAlphabetResolved = in.readBoolean();
		dataDictClass = (Class) in.readObject();
		targetDictClass = (Class) in.readObject();
	}

	
}
