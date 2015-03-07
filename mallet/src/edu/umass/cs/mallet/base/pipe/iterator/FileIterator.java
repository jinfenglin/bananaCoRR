/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.pipe.iterator;

import edu.umass.cs.mallet.base.pipe.Pipe;
import edu.umass.cs.mallet.base.types.Label;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Alphabet;
import java.util.ArrayList;
import java.util.Iterator;
import java.net.URI;
import java.util.regex.*;
import java.io.*;

public class FileIterator extends AbstractPipeInputIterator
{
	FileFilter fileFilter;
	ArrayList fileArray;
	Iterator subIterator;
	Pattern targetPattern;								// Set target slot to string coming from 1st group of this Pattern
	File[] startingDirectories;
	int[] minFileIndex;
	int fileCount;

	/** Special value that means to use the directories[i].getPath() as the target name */
	// xxx Note that these are specific to UNIX directory delimiter characters!  Fix this.
	public static final Pattern STARTING_DIRECTORIES = Pattern.compile ("_STARTING_DIRECTORIES_");
	public static final Pattern FIRST_DIRECTORY = Pattern.compile ("/?([^/]*)/.+");
	public static final Pattern LAST_DIRECTORY = Pattern.compile ("([^/]*)/[^/]+");
	public static final Pattern ALL_DIRECTORIES = Pattern.compile ("^(.*)/[^/]+");


	// added by Fuchun Peng	
	public ArrayList getFileArray()
	{
		return fileArray;
	}

	/** Pass null as targetPattern to get null targets */
	protected FileIterator (File[] directories, FileFilter fileFilter,
													Pattern targetPattern)
	{
		this.startingDirectories = directories;
		this.fileFilter = fileFilter;
		this.minFileIndex = new int[directories.length];
		this.fileArray = new ArrayList ();
		this.targetPattern = targetPattern;
		for (int i = 0; i < directories.length; i++) {
			if (!directories[i].isDirectory())
				throw new IllegalArgumentException (directories[i].getAbsolutePath()
																					+ " is not a directory.");
			minFileIndex[i] = fileArray.size();
			fillFileArray (directories[i], fileFilter, fileArray);
		}
		this.subIterator = fileArray.iterator();
		this.fileCount = 0;

		//print the files
//		for(int i=0; i<fileArray.size(); i++){
//			File file = (File) fileArray.get(i);
//			System.out.println(file.toString());
//		}

	}

	/** Iterate over Files that pass the fileFilter test, setting... */
	public FileIterator (File[] directories, Pattern targetPattern)
	{
		this (directories, null, targetPattern);
	}

	public static File[] stringArray2FileArray (String[] sa)
	{
		File[] ret = new File[sa.length];
		for (int i = 0; i < sa.length; i++)
			ret[i] = new File (sa[i]);
		return ret;
	}

	public FileIterator (String[] directories, FileFilter ff)
	{
		this (stringArray2FileArray(directories), ff, null);
	}
	
	public FileIterator (String[] directories, String targetPattern)
	{
		this (stringArray2FileArray(directories), Pattern.compile(targetPattern));
	}

	public FileIterator (String[] directories, Pattern targetPattern)
	{
		this (stringArray2FileArray(directories), targetPattern);
	}
	
	public FileIterator (File directory, FileFilter fileFilter, Pattern targetPattern)
	{
		this (new File[] {directory}, fileFilter, targetPattern);
	}

	public FileIterator (File directory, FileFilter fileFilter)
	{
		this (new File[] {directory}, fileFilter, null);
	}
	
	public FileIterator (File directory, Pattern targetPattern)
	{
		this (new File[] {directory}, null, targetPattern);
	}

	public FileIterator (String directory, Pattern targetPattern)
	{
		this (new File[] {new File(directory)}, null, targetPattern);
	}
	
	public FileIterator (File directory)
	{
		this (new File[] {directory}, null, null);
	}

	public FileIterator (String directory)
	{
		this (new File[] {new File(directory)}, null, null);
	}
	
	
	private int fillFileArray (File directory, FileFilter filter, ArrayList files)
	{
		int count = 0;
		File[] directoryContents = directory.listFiles();
		for (int i = 0; i < directoryContents.length; i++) {
			if (directoryContents[i].isDirectory())
				count += fillFileArray (directoryContents[i], filter, files);
			else if (filter == null || filter.accept(directoryContents[i])) {
				files.add (directoryContents[i]);
				count++;
			}
		}
		return count;
	}

	// The PipeInputIterator interface
	public Instance nextInstance ()
	{
		File nextFile = (File) subIterator.next();
		String path = nextFile.getAbsolutePath();
		String targetName = null;
		if (targetPattern == STARTING_DIRECTORIES) {
			int i;
			for (i = 0; i < minFileIndex.length; i++)
				if (minFileIndex[i] > fileCount)
					break;
			targetName = startingDirectories[--i].getPath();
		} else if (targetPattern != null) {
			Matcher m = targetPattern.matcher(path);
			if (m.find ())
				targetName = m.group (1);
		}
		fileCount++;
		return new Instance (nextFile, targetName, nextFile.toURI(), null);
	}

	// culotta - 9.11.03
	public File nextFile ()
	{
		return (File) subIterator.next();		
	}

	public boolean hasNext ()	{	return subIterator.hasNext();	}
	
}

