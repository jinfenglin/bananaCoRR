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

import java.util.ArrayList;
import java.io.*;

public class Alphabet implements Serializable
{
	gnu.trove.TObjectIntHashMap map;
	ArrayList entries;
	boolean growthStopped = false;
	Class entryClass = null;

	public Alphabet (int capacity, Class entryClass)
	{
		this.map = new gnu.trove.TObjectIntHashMap (capacity);
		this.entries = new ArrayList (capacity);
		this.entryClass = entryClass;
	}

	public Alphabet (Class entryClass)
	{
		this (8, entryClass);
	}
	
	public Alphabet (int capacity)
	{
		this (capacity, null);
	}

	public Alphabet ()
	{
		this (8, null);
	}

	public Object clone () 
	{
		//try {
		// Wastes effort, because we over-write ivars we create
		Alphabet ret = new Alphabet ();
		ret.map = (gnu.trove.TObjectIntHashMap) map.clone();
		ret.entries = (ArrayList) entries.clone();
		ret.growthStopped = growthStopped;
		ret.entryClass = entryClass;
		return ret;
		//} catch (CloneNotSupportedException e) {
		//e.printStackTrace();
		//throw new IllegalStateException ("Couldn't clone InstanceList Vocabuary");
		//}
	}
	
	/** Return -1 if entry isn't present. */
	public int lookupIndex (Object entry, boolean addIfNotPresent)
	{
		if (entry == null)
			throw new IllegalArgumentException ("Can't lookup \"null\" in an Alphabet.");
		if (entryClass == null)
			entryClass = entry.getClass();
		else
			// Insist that all entries in the Alphabet are of the same
			// class.  This may not be strictly necessary, but will catch a
			// bunch of easily-made errors.
			if (entry.getClass() != entryClass)
				throw new IllegalArgumentException ("Non-matching entry class, "+entry.getClass()+", was "+entryClass);
		int ret = map.get(entry);
		if (ret == -1 && !growthStopped && addIfNotPresent) {
			if (entry instanceof String)
				entry = ((String)entry).intern();
			ret = entries.size();
			map.put (entry, entries.size());
			entries.add (entry);
		}
		return ret;
	}

	public int lookupIndex (Object entry)
	{
		return lookupIndex (entry, true);
	}
	
	public Object lookupObject (int index)
	{
		return entries.get(index);
	}

	public Object[] lookupObjects (int[] indices)
	{
		Object[] ret = new Object[indices.length];
		for (int i = 0; i < indices.length; i++)
			ret[i] = entries.get(indices[i]);
		return ret;
	}

	public int[] lookupIndices (Object[] objects, boolean addIfNotPresent)
	{
		int[] ret = new int[objects.length];
		for (int i = 0; i < objects.length; i++)
			ret[i] = lookupIndex (objects[i], addIfNotPresent);
		return ret;
	}

	public boolean contains (Object entry)
	{
		return map.contains (entry);
	}

	public int size ()
	{
		return entries.size();
	}

	public void stopGrowth ()
	{
		growthStopped = true;
	}

	public boolean growthStopped ()
	{
		return growthStopped;
	}

	public Class entryClass ()
	{
		return entryClass;
	}

	// Serialization 
		
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeInt (entries.size());
		for (int i = 0; i < entries.size(); i++)
			out.writeObject (entries.get(i));
		out.writeBoolean (growthStopped);
		out.writeObject (entryClass);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		int size = in.readInt();
		entries = new ArrayList (size);
		map = new gnu.trove.TObjectIntHashMap (size);
		for (int i = 0; i < size; i++) {
			Object o = in.readObject();
			map.put (o, i);
			entries. add (o);
		}
		growthStopped = in.readBoolean();
		entryClass = (Class) in.readObject();
	}
	
}
