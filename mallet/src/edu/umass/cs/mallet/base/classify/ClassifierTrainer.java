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

import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.InstanceList;
import edu.umass.cs.mallet.base.types.FeatureSelection;
import edu.umass.cs.mallet.base.classify.Classifier;
import edu.umass.cs.mallet.base.util.CommandOption;
import edu.umass.cs.mallet.base.util.BshInterpreter;
import java.io.*;
import java.util.*;

public abstract class ClassifierTrainer
{
	public Classifier train (InstanceList trainingSet)
	{
		return this.train (trainingSet, null);
	}

	public Classifier train (InstanceList trainingSet,
													 InstanceList validationSet)
	{
		return this.train (trainingSet, validationSet, null);
	}

	public Classifier train (InstanceList trainingSet,
													 InstanceList validationSet,
													 InstanceList testSet)
	{
		return this.train (trainingSet, validationSet, testSet, null, null);
	}

	public Classifier train (InstanceList trainingSet,
													 InstanceList validationSet,
													 InstanceList testSet,
													 ClassifierEvaluating evaluator)
	{
		return this.train (trainingSet, validationSet, testSet, evaluator, null);
	}
	
	/** Return a new classifier tuned using the three arguments.
			@param trainingSet examples used to set parameters.
			@param validationSet examples used to tune meta-parameters.  May be null.
			@param testSet examples not examined at all for training, but passed on to diagnostic routines.  May be null.
			@param initialClassifier training process may start from here.  The parameters of the initialClassifier are not modified.  May be null.
	*/
	public abstract Classifier train (InstanceList trainingSet,
																		InstanceList validationSet,
																		InstanceList testSet,
																		ClassifierEvaluating evaluator,
																		Classifier initialClassifier);

	public String toString()
	{
		return this.getClass().getName();
	}


	// Command-line interface

	static CommandOption.Object trainerConstructorOption = new CommandOption.Object
	(ClassifierTrainer.class, "trainer", "ClassifierTrainer constructor",	true, new NaiveBayesTrainer(),
	 "Java code for the constructor for the ClassifierTrainer to use", null);

//		{ public void postParsing (CommandOption.List list) {
//			java.lang.String classname = value.split ("[^A-Za-z0-9]*", 2)[0];
//			try {
//				Class trainerClass = Class.forName(classname);
//				System.out.println ("Trying to add CommandOptions for class "+trainerClass.getName());
//				if (CommandOption.ListProviding.class.isAssignableFrom(trainerClass))
//					;
//				//list.add (trainerClass.getCommandOptionList());
//			} catch (ClassNotFoundException e) {
//				System.out.println ("Couldn't find class "+classname);
//				e.printStackTrace();
//				throw new IllegalArgumentException ("Couldn't find class "+classname);
//			}
//		}};

	static CommandOption.String outputFilenameOption = new CommandOption.String
	(ClassifierTrainer.class, "output-file", "FILENAME", true, "classifier.mallet",
	 "The filename in which to write the resulting classifier.", null);

	static CommandOption.String instanceListFilenameOption = new CommandOption.String
	(ClassifierTrainer.class, "instance-list", "FILENAME", true, null,
	 "The filename from which to read the list of training instances.  "+
	 "Default is not to write the classifier to disk", null);

	static CommandOption.Double trainingProportionOption = new CommandOption.Double
	(ClassifierTrainer.class, "training-proportion", "DECIMAL", true, 1.0,
	 "The fraction of the instances that should be used for training.", null);

	static CommandOption.Double validationProportionOption = new CommandOption.Double
	(ClassifierTrainer.class, "validation-proportion", "DECIMAL", true, 0.0,
	 "The fraction of the instances that should be used for validation.", null);
	
	static CommandOption.Integer randomSeedOption = new CommandOption.Integer
	(ClassifierTrainer.class, "random-seed", "INTEGER", true, 0,
	 "The random seed for randomly selecting a proportion of the instance list for training", null);

	static CommandOption.Object classifierEvaluatorOption = new CommandOption.Object
	(ClassifierTrainer.class, "classifier-evaluator", "CONSTRUCTOR", true, null,
	 "Java code for constructing a ClassifierEvaluating object", null);
	
	static CommandOption.Boolean printTrainAccuracyOption = new CommandOption.Boolean
	(ClassifierTrainer.class, "print-train-accuracy", "true|false", true, true,
	 "After training, run the resulting classifier on the instances included in training, "
	 +"and print the accuracy", null);

	static CommandOption.Boolean printTestAccuracyOption = new CommandOption.Boolean
	(ClassifierTrainer.class, "print-test-accuracy", "true|false", true, true,
	 "After training, run the resulting classifier on the instances not included in training, "
	 +"and print the accuracy", null);

	static final CommandOption.List commandOptions =
	new CommandOption.List (
		"Training a classifier, printing accuracies and saveing the trained classifier",
		new CommandOption[] {
			trainerConstructorOption,
			outputFilenameOption,
			instanceListFilenameOption,
			trainingProportionOption,
			randomSeedOption,
			printTrainAccuracyOption,
			printTestAccuracyOption,
		});

	public static void main (String[] args) throws bsh.EvalError, java.io.IOException
	{
		// Process the command-line options
		commandOptions.process (args);

		System.out.println ("Trainer = "+trainerConstructorOption.value.toString());
		ClassifierTrainer trainer = (ClassifierTrainer) trainerConstructorOption.value;
		InstanceList ilist = InstanceList.load (new File(instanceListFilenameOption.value));

		Random r = randomSeedOption.wasInvoked() ? new Random (randomSeedOption.value) : new Random ();
		double t = trainingProportionOption.value;
		double v = validationProportionOption.value;
		InstanceList[] ilists = ilist.split (r, new double[] {t, v, 1-t-v});
		System.err.println ("Training...");
		Classifier c = trainer.train (ilists[0], ilists[1], null,
																	(ClassifierEvaluating)classifierEvaluatorOption.value, null);
		if (printTrainAccuracyOption.value)
			System.out.print ("Train accuracy = " + c.getAccuracy(ilists[0]) + "  ");
		if (printTestAccuracyOption.value)
			System.out.print ("Test accuracy  = " + c.getAccuracy(ilists[2]));
		if (printTrainAccuracyOption.value || printTestAccuracyOption.value)
			System.out.println ("");
		if (outputFilenameOption.wasInvoked()) {
			try {
				ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream (instanceListFilenameOption.value));
				oos.writeObject (c);
				oos.close();
			} catch (Exception e) {
				e.printStackTrace();
				throw new IllegalArgumentException ("Couldn't write classifier to filename "+
					instanceListFilenameOption.value);
			}
		}
	}
	
}
