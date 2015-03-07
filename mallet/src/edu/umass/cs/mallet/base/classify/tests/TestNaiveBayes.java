/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.classify.tests;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.classify.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.util.*;
import edu.umass.cs.mallet.base.pipe.iterator.ArrayIterator;
import junit.framework.*;
import java.net.URI;

public class TestNaiveBayes extends TestCase
{
	public TestNaiveBayes (String name)
	{
		super (name);
	}

	public void testNonTrained ()
	{
		Alphabet fdict = new Alphabet ();
		System.out.println ("fdict.size="+fdict.size());
		LabelAlphabet ldict = new LabelAlphabet ();
		Multinomial.Estimator me1 = new Multinomial.LaplaceEstimator (fdict);
		Multinomial.Estimator me2 = new Multinomial.LaplaceEstimator (fdict);

		// Prior
		ldict.lookupIndex ("sports");
		ldict.lookupIndex ("politics");
		ldict.stopGrowth ();
		System.out.println ("ldict.size="+ldict.size());
		Multinomial prior = new Multinomial (new double[] {.5, .5}, ldict);
		
		// Sports
		me1.increment ("win", 5);
		me1.increment ("puck", 5);
		me1.increment ("team", 5);
		System.out.println ("fdict.size="+fdict.size());

		// Politics
		me2.increment ("win", 5);
		me2.increment ("speech", 5);
		me2.increment ("vote", 5);

		Multinomial sports = me1.estimate();
		Multinomial politics = me2.estimate();

		// We must estimate from me1 and me2 after all data is incremented,
		// so that the "sports" multinomial knows the full dictionary size!

		Classifier c = new NaiveBayes (new Noop (fdict, ldict),
																	 prior,
																	 new Multinomial[] {sports, politics});

		Instance inst =
			new Instance (new FeatureVector (fdict,
																			 new Object[] {"speech", "win"},
																			 new double[] {1, 1}),
										ldict.lookupLabel ("politics"),
										null, null);
		System.out.println ("inst.data = "+inst.getData ());
		
		Classification cf = c.classify (inst);
		LabelVector l = (LabelVector) cf.getLabeling();
		//System.out.println ("l.size="+l.size());
		System.out.println ("l.getBestIndex="+l.getBestIndex());
		assertTrue (cf.getLabeling().getBestLabel()
								== ldict.lookupLabel("politics"));
		assertTrue (cf.getLabeling().getBestValue()	> 0.6);
	}

	public void testStringTrained ()
	{
		String[] africaTraining = new String[] {
			"on the plains of africa the lions roar",
			"in swahili ngoma means to dance",
			"nelson mandela became president of south africa",
			"the saraha dessert is expanding"};
		String[] asiaTraining = new String[] {
			"panda bears eat bamboo",
			"china's one child policy has resulted in a surplus of boys",
			"tigers live in the jungle"};

		InstanceList instances =
			new InstanceList (
				new SerialPipes (new Pipe[] {
					new Target2Label (),
					new CharSequence2TokenSequence (),
					new TokenSequence2FeatureSequence (),
					new FeatureSequence2FeatureVector ()}));

		instances.add (new ArrayIterator (africaTraining, "africa"));
		instances.add (new ArrayIterator (asiaTraining, "asia"));
		Classifier c = new NaiveBayesTrainer ().train (instances);

		Classification cf = c.classify ("nelson mandela never eats lions");
		assertTrue (cf.getLabeling().getBestLabel()
								== ((LabelAlphabet)instances.getTargetAlphabet()).lookupLabel("africa"));
	}

	public void testRandomTrained ()
	{
		InstanceList ilist = new InstanceList (new Random(1), 10, 2);
		Classifier c = new NaiveBayesTrainer ().train (ilist);
		// test on the training data
		int numCorrect = 0;
		for (int i = 0; i < ilist.size(); i++) {
			Instance inst = ilist.getInstance(i);
			Classification cf = c.classify (inst);
			cf.print ();
			if (cf.getLabeling().getBestLabel() == inst.getLabeling().getBestLabel())
				numCorrect++;
		}
		System.out.println ("Accuracy on training set = " + ((double)numCorrect)/ilist.size());
	}
	
	
	public static Test suite ()
	{
		return new TestSuite (TestNaiveBayes.class);
	}

	protected void setUp ()
	{
	}

	public static void main (String[] args)
	{
		junit.textui.TestRunner.run (suite());
	}
	
}
