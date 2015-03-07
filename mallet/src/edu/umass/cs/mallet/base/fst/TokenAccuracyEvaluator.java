/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.fst;

import edu.umass.cs.mallet.base.types.*;
import edu.umass.cs.mallet.base.util.MalletLogger;

import java.util.logging.*;
import java.io.*;

public class TokenAccuracyEvaluator extends TransducerEvaluator
{
	private static Logger logger = MalletLogger.getLogger(TokenAccuracyEvaluator.class.getName());

	public TokenAccuracyEvaluator (boolean printViterbiPath)
	{
		viterbiOutput = printViterbiPath;
	}

	public TokenAccuracyEvaluator ()
	{
		this (false);
	}
	
	public boolean evaluate (Transducer crf, boolean finishedTraining, int iteration,
													 boolean converged, double cost,
													 InstanceList training, InstanceList validation, InstanceList testing)
	{
		logger.info ("Iteration="+iteration+" Cost="+cost);
		InstanceList[] lists = new InstanceList[] {training, validation, testing};
		String[] listnames = new String[] {"Training", "Validation", "Testing"};
		for (int k = 0; k < lists.length; k++)
			if (lists[k] != null)
        test(crf, lists[k], listnames[k], null);
		return true;
	}

  public void test(Transducer model, InstanceList data, String description,
                   PrintStream viterbiOutputStream)
  {
		int numCorrectTokens;
		int totalTokens;

    totalTokens = numCorrectTokens = 0;
    logger.info ("Results for "+description);
    for (int i = 0; i < data.size(); i++) {
      Instance instance = data.getInstance(i);
      Sequence input = (Sequence) instance.getData();
      Sequence trueOutput = (Sequence) instance.getTarget();
      assert (input.size() == trueOutput.size());
      Sequence predOutput = model.viterbiPath(input).output();
      assert (predOutput.size() == trueOutput.size());
      for (int j = 0; j < trueOutput.size(); j++) {
        totalTokens++;
        if (trueOutput.get(j).equals(predOutput.get(j)))
          numCorrectTokens++;
        if (viterbiOutputStream != null) {
          FeatureVector fv = (FeatureVector) input.get(j);
          viterbiOutputStream.println(trueOutput.get(j).toString()+'/'+predOutput.get(j).toString()+"  "+
           fv.toString(true));
        }
      }
    }
			logger.info (description +" accuracy="+((double)numCorrectTokens)/totalTokens);
  }
	
}
