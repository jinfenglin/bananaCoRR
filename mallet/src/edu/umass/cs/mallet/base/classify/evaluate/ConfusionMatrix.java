/* Copyright (C) 2002 Dept. of Computer Science, Univ. of Massachusetts, Amherst

   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet

   This program toolkit free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  For more
   details see the GNU General Public License and the file README-LEGAL.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
   02111-1307, USA. */


/** 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.umass.cs.mallet.base.classify.evaluate;

import edu.umass.cs.mallet.base.classify.Classification;
import edu.umass.cs.mallet.base.classify.Trial;
import edu.umass.cs.mallet.base.types.Labeling;
import edu.umass.cs.mallet.base.types.LabelVector;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Label;
import edu.umass.cs.mallet.base.util.MalletLogger;

import java.util.ArrayList;
import java.util.logging.*;
import java.text.*;

/**
 * Calculates and prints confusion matrix, accuracy,
 * and precision for a given clasification trial.
 */
public class ConfusionMatrix
{
	private static Logger logger = MalletLogger.getLogger(ConfusionMatrix.class.getName());
	
	int NUM_CLASSES;
  /**
   * the list of classifications from the trial
   */
	ArrayList classifications;
	/**
	 * 2-d confiusion matrix
	 */
	int[][] values;

	/**
	 * Constructs matrix and calculates values
	 * @param t the trial to build matrix from
	 */
	public ConfusionMatrix(Trial t)
	{
		this.classifications = t.toArrayList();
		Labeling tempLabeling =
			((Classification)classifications.get(0)).getLabeling();
		this.NUM_CLASSES = tempLabeling.getLabelAlphabet().size();
		logger.info("Num classes = " + this.NUM_CLASSES);
		values = new int[NUM_CLASSES][NUM_CLASSES];
		initValues();
		fillValues();
	}

	/**
	 * Initizlize values to 0
	 */
	private void initValues()
	{
		for(int i=0; i<this.NUM_CLASSES; i++)
			for(int j=0; j<this.NUM_CLASSES; j++)
				values[i][j]=0;
	}
	
        /** 
         * Return value at i,j
         */
        public double value(int i, int j) 
        {
	    assert(i >= 0 && j >= 0 && i < NUM_CLASSES && j < NUM_CLASSES);
	    return values[i][j];	    
        }
	/**
	 * Fills the matrix with appropriate values
	 */
	private void fillValues()
	{
		for(int i=0; i < classifications.size(); i++)
		{
			LabelVector lv =
				((Classification)classifications.get(i)).getLabelVector();
			Instance inst = ((Classification)classifications.get(i)).getInstance();
			int bestIndex = lv.getBestIndex();
			int correctIndex = inst.getLabeling().getBestIndex();
			assert(correctIndex != -1);
			//System.out.println("Best index="+bestIndex+". Correct="+correctIndex);
			values[bestIndex][correctIndex]++;
		}			
	}	
	
	/**
	 * Returns the precision of this predicted class
	 */
	public double getPrecision (int predictedClassIndex)
	{
		int total = 0;
		for (int trueClassIndex=0; trueClassIndex < this.NUM_CLASSES; trueClassIndex++) {
			total += values[predictedClassIndex][trueClassIndex];
		}
		if (total == 0)
			return 0.0;
		else
			return (double) (values[predictedClassIndex][predictedClassIndex]) / total;
	}
	
	/**
	 * Returns percent of time that class2 is true class when 
	 * class1 is predicted class
	 * 
	 */
	public double getConfusionBetween (int class1, int class2)
	{
		int total = 0;
		for (int trueClassIndex=0; trueClassIndex < this.NUM_CLASSES; trueClassIndex++) {
			total += values[class1][trueClassIndex];
		}
		if (total == 0)
			return 0.0;
		else
			return (double) (values[class1][class2]) / total;	    
	}

	/**
	 * Returns the percentage of instances with
	 * true label = classIndex
	 */
	public double getClassPrior (int classIndex)
	{
		int sum= 0;
		for(int i=0; i < NUM_CLASSES; i++) 
			sum += values[i][classIndex];
		return (double)sum / classifications.size();
	}

  /**
	 * prints to stdout the confusion matrix,
	 * class frequency, precision, and recall
	 */
	public void print()
	{
		LabelVector lv =
			((Classification)classifications.get(0)).getLabelVector();
		DecimalFormat df = new DecimalFormat("###.##");
		int [] numInstances = new int[this.NUM_CLASSES];
		for(int i=0; i<this.NUM_CLASSES; i++){
			int count = 0;
			for(int j=0; j<this.NUM_CLASSES; j++)
				count += values[j][i];
			numInstances[i] = count;
			String label = lv.labelAtLocation(i).toString();
			System.out.println("index "+i+": "+label+
												 " "+count+" instances "+
												 df.format(100*(double)count/classifications.size())
				+"%");
		}
		System.out.println("Confusion Matrix");
		for(int i=0; i<this.NUM_CLASSES; i++){
			for(int j=0; j<this.NUM_CLASSES; j++)
				System.out.print(values[i][j]+"\t\t");
			System.out.println("");
		}
		for(int i=0; i<this.NUM_CLASSES; i++){
			double recall = 100.0*(double)values[i][i]/numInstances[i];
			double precision;
			int rowCount = 0;
			for(int j=0; j<this.NUM_CLASSES; j++)
				rowCount += values[i][j];
			precision = 100.0*(double)values[i][i] / rowCount;
			System.out.println("For class " + i);
			System.out.println("Recall="+df.format(recall)+"%");
			System.out.println("Precision="+df.format(precision)+"%");
		}
		
		int numCorrect = 0;
		int totalInstances = 0;
		for(int i=0; i<this.NUM_CLASSES; i++)
		{
			numCorrect += values[i][i];
			totalInstances+=numInstances[i];
		}
		System.out.println("Overall Accuracy="+
											 df.format(100.0*(double)numCorrect/totalInstances)+"%");
	}
}















