/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.classify;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.classify.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.pipe.iterator.*;
import edu.umass.cs.mallet.base.util.*;
import java.io.*;

public class TUI
{
	static CommandOption.File instanceListFile = new CommandOption.File
	(TUI.class, "instance-list", "FILE", true, new File("instance-list.mallet"),
	 "Read or write the instance list to this file.", null);

	static CommandOption.Set whatDoing = new CommandOption.Set
	(TUI.class, "mode", "MODENAME", true, new String[] {"index", "test"}, 0,
	 "Set primary option.", null);

	static CommandOption.SpacedStrings indexTextFileDirectories =	new CommandOption.SpacedStrings
	(TUI.class, "index-text-dirs", "DIR...", true, null,
	 "The directories containing text files to be classified, one directory per class", null);

	static CommandOption.SpacedStrings indexCsvLines =	new CommandOption.SpacedStrings
	(TUI.class, "index-csv-lines", "FILENAME", true, new String[] {"-"},
	 "The name of the file containing one line per instance to be classified", null);
	
	public static void main (String[] args)
	{
		if (args.length == 0)
			args = new String[] {
				"/usr/dan/users11/culotta/work/data/20news-18828/comp.graphics",
				"/usr/dan/users11/culotta/work/data/20news-18828/comp.os.ms-windows.misc",
				"/usr/dan/users11/culotta/work/data/20news-18828/comp.sys.ibm.pc.hardware",
				"/usr/dan/users11/culotta/work/data/20news-18828/comp.sys.mac.hardware",
				"/usr/dan/users11/culotta/work/data/20news-18828/comp.windows.x",
				};

		File[] directories = new File[args.length];
		for (int i = 0; i < args.length; i++)
			directories[i] = new File (args[i]);
		Pipe instancePipe = new SerialPipes (new Pipe[] {
			new Target2Label (),
			new Input2CharSequence (),
			new CharSubsequence (CharSubsequence.SKIP_HEADER),
			new CharSequence2TokenSequence (),
			new TokenSequenceLowercase (),
			new TokenSequenceRemoveStopwords (),
			new TokenSequence2FeatureSequence(),
			//new PrintInputAndTarget (),
			//new FeatureSequence2FeatureVector(),
			new FeatureSequence2AugmentableFeatureVector(),
			//new PrintInputAndTarget ()
		});
		InstanceList ilist = new InstanceList (instancePipe);
		ilist.add (new FileIterator (directories, FileIterator.STARTING_DIRECTORIES));

		java.util.Random r = new java.util.Random (1);
		InstanceList[] ilists = ilist.split (r, new double[] {.3, .7});

		InfoGain ig = new InfoGain (ilists[0]);
		for (int i = 0; i < 10; i++)
			System.out.println ("InfoGain["+ig.getObjectAtRank(i)+"]="+ig.getValueAtRank(i));
		//ig.print();
		
		FeatureSelection selectedFeatures = new FeatureSelection (ig, 8000);
		ilists[0].setFeatureSelection (selectedFeatures);
		//OddsRatioFeatureInducer orfi = new OddsRatioFeatureInducer (ilists[0]);
		//orfi.induceFeatures (ilists[0], false, true);

		System.out.println ("Training with "+ilists[0].size()+" instances");
		ClassifierTrainer[] trainers = new ClassifierTrainer[] {
			//new NaiveBayesTrainer(),
			new MaxEntTrainer(),
			//new FeatureSelectingClassiferTrainer (new MaxEntTrainer(),
			//new PerClassInfoGainFeatureSelector (200)),
			//new DecisionTreeTrainer(4),
			//new RoostingTrainer(),
			//new AdaBoostTrainer(new DecisionTreeTrainer(2),10),
		};
		double accuracies[] = new double[trainers.length];
		Classifier classifiers[] = new Classifier[trainers.length];
		long time[] = new long[trainers.length];
		for (int c = 0; c < classifiers.length; c++){
                    time[c] = System.currentTimeMillis();
		    classifiers[c] = trainers[c].train (ilists[0]);
		    time[c] = System.currentTimeMillis() - time[c];
		}
		for (int c = 0; c < classifiers.length; c++) {
		    System.out.println (trainers[c].toString()
					+" training accuracy = "+ new Trial (classifiers[c], ilists[0]).accuracy()+"\nTime(sec):"+(time[c]/1000.0));
		    System.out.println (trainers[c].toString()
					+" testing accuracy = "+ new Trial (classifiers[c], ilists[1]).accuracy());
		}
	}
    
}
