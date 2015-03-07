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

import edu.umass.cs.mallet.base.types.InstanceList;
import java.io.*;

public abstract class TransducerEvaluator
{
  protected int numIterationsToWait = 0;
	protected int numIterationsToSkip = 0;
	protected boolean alwaysEvaluateWhenFinished = true;
	protected boolean printModelAtEnd = false;
	protected boolean checkpointTransducer = false;
	protected String checkpointFilePrefix = null;
	protected int checkpointIterationsToSkip = 9;
  protected	boolean viterbiOutput = true;
	protected String viterbiOutputFilePrefix = null;
	protected int viterbiOutputIterationsToWait = 10;
	protected int viterbiOutputIterationsToSkip = 10;
	protected String viterbiOutputEncoding = "UTF-8";

  public void setViterbiOutput(boolean vo)
  {
    viterbiOutput = vo;
  }

  public boolean getViterbiOutput()
  {
    return viterbiOutput;
  }

  public void setViterbiOutputFilePrefix(String p)
  {
    viterbiOutputFilePrefix = p;
  }

  public String getViterbiOutputFilePrefix()
  {
    return viterbiOutputFilePrefix;
  }

  public void setViterbiOutputIterationsToWait(int i)
  {
    viterbiOutputIterationsToWait = i;
  }

  public int getViterbiOutputIterationsToWait()
  {
    return viterbiOutputIterationsToWait;
  }

  public void setViterbiOutputIterationsToSkip(int i)
  {
    viterbiOutputIterationsToSkip = i;
  }

  public int getViterbiOutputIterationsToSkip()
  {
    return viterbiOutputIterationsToSkip;
  }

  public void setViterbiOutputEncoding(String o)
  {
    viterbiOutputEncoding = o;
  }

  public String getViterbiOutputEncoding()
  {
    return viterbiOutputEncoding;
  }
	
  public void setCheckpointTransducer(boolean c)
  {
    checkpointTransducer = c;
  }

  public boolean getCheckpointTransducer()
  {
    return checkpointTransducer;
  }

  public void setCheckpointFilePrefix(String p)
  {
    checkpointFilePrefix = p;
  }

  public String getCheckpointFilePrefix()
  {
    return checkpointFilePrefix;
  }

  public void setCheckpointIterationsToSkip(int i)
  {
    checkpointIterationsToSkip = i;
  }

  public int getCheckpointIterationsToSkip()
  {
    return checkpointIterationsToSkip;
  }  

	/** Training will terminate if "false" is returned. */
	public abstract boolean evaluate (Transducer transducer,
                                    boolean finishedTraining, int iteration,
                                    boolean converged, double cost,
                                    InstanceList training,
                                    InstanceList validation,
                                    InstanceList testing);
  public abstract void test(Transducer transducer, InstanceList data,
                   String description, PrintStream viterbiOutputStream);
                   
}
