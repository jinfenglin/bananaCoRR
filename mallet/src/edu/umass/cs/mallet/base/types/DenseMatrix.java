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

import java.io.*;

public abstract class DenseMatrix implements Matrix, Serializable
{
	double[] values;
	
	public abstract int getNumDimensions ();
	public abstract int getDimensions (int[] sizes);
	
	public abstract double value (int[] indices);
	public abstract void setValue (int[] indices, double value);
	public abstract ConstantMatrix cloneMatrix ();

	public abstract int singleIndex (int[] indices);
	public abstract void singleToIndices (int i, int[] indices);
	public double singleValue (int i) { return values[i]; }
	public void setSingleValue (int i, double value) { values[i] = value; }
	public void incrementSingleValue (int i, double delta) { values[i] += delta; }
	public int singleSize () { return values.length; }

	public int numLocations () { return values.length; }
	public int location (int index) { return index; }
	public double valueAtLocation (int location) { return values[location]; }
	// Returns a "singleIndex"
	public int indexAtLocation (int location) { return location; }

	
	public void setAll (double v) { for (int i = 0; i < values.length; i++) values[i] = v; }
	public void set (ConstantMatrix m) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			System.arraycopy (((DenseMatrix)m).values, 0, values, 0, values.length);
		} else
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] = m.valueAtLocation(i);
	}
	
	public void setWithAddend (ConstantMatrix m, double addend) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				values[i] = ((DenseMatrix)m).values[i] + addend;
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] = m.valueAtLocation(i) + addend;
	}
		
	public void setWithFactor (ConstantMatrix m, double factor) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				values[i] = ((DenseMatrix)m).values[i] * factor;
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] = m.valueAtLocation(i) * factor;
	}

	public void plusEquals (double v) {
		for (int i = 0; i < values.length; i++)
			values[i] += v;
	}
	
	public void plusEquals (ConstantMatrix m) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++) {
				// added by Culotta - 12.10.0 to enforce INF - INF = 0
				if(Double.isInfinite(values[i]) &&
					 Double.isInfinite(((DenseMatrix)m).values[i])) {
					double newValue = ((DenseMatrix)m).values[i];
					// make sure they're opposite signed 
					if((newValue * values[i]) < 0) {
						values[i] = 0.0; // inf - inf = 0
					}
					else
						values[i] += newValue;
				}
				else 
					values[i] += ((DenseMatrix)m).values[i];
			}
		}
		else  
			for (int i = m.numLocations()-1; i >= 0; i--) {
				// added by Culotta - 12.10.02 to enforce INF - INF = 0
				if(Double.isInfinite(values[m.indexAtLocation(i)]) &&
					 Double.isInfinite(((DenseMatrix)m).values[i])) {
					double newValue = m.valueAtLocation(i);
					// make sure they're oppisite signed 
					if((newValue * values[m.indexAtLocation(i)]) < 0) {
						values[m.indexAtLocation(i)] = 0.0;
					}
					else
						values[m.indexAtLocation(i)] += newValue;
				}
				else
					values[m.indexAtLocation(i)] += m.valueAtLocation(i);
			}
	}

	public void plusEquals (ConstantMatrix m, double factor) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++) {
				// added by Culotta - 12.10.0 to enforce INF - INF = 0
				if(Double.isInfinite(values[i]) &&
					 Double.isInfinite(((DenseMatrix)m).values[i])) {
					double newValue = factor*((DenseMatrix)m).values[i];
					// make sure they're opposite signed 
					if((newValue * values[i]) < 0) {
						values[i] = 0.0; // inf - inf = 0
					}
					else
						values[i] += newValue;
				}
				else
					values[i] += ((DenseMatrix)m).values[i] * factor;
			}
		}
		else 
			for (int i = m.numLocations()-1; i >= 0; i--){
				// added by Culotta - 12.10.02 to enforce INF - INF = 0
				if(Double.isInfinite(values[m.indexAtLocation(i)]) &&
					 Double.isInfinite(((DenseMatrix)m).values[i])) {
					double newValue = factor*m.valueAtLocation(i);
					// make sure they're oppisite signed 
					if((newValue * values[m.indexAtLocation(i)]) < 0) {
						values[m.indexAtLocation(i)] = 0.0;
					}
					else
						values[m.indexAtLocation(i)] += newValue;
				}
				else
					values[m.indexAtLocation(i)] += m.valueAtLocation(i) * factor;
			}
	}

	public void equalsPlus (double factor, ConstantMatrix m) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++) {
				// added by Culotta - 12.10.0 to enforce INF - INF = 0
				if(Double.isInfinite(values[i]) &&
					 Double.isInfinite(((DenseMatrix)m).values[i])) {
					double lhs = factor*values[i];
					double rhs = ((DenseMatrix)m).values[i];
					// make sure they're opposite signed 
					if((lhs * rhs) < 0) {
						values[i] = 0.0; // inf - inf = 0
					}
					else
						values[i] = lhs + rhs;
				}
				else
					values[i] = factor*values[i] + ((DenseMatrix)m).values[i];
			}
		}
		else 
			for (int i = m.numLocations()-1; i >= 0; i--) {				
				// added by Culotta - 12.10.02 to enforce INF - INF = 0
				if(Double.isInfinite(values[m.indexAtLocation(i)]) &&
					 Double.isInfinite(((DenseMatrix)m).values[i])) {
					double lhs = factor * values[m.indexAtLocation(i)];
					double rhs = m.valueAtLocation(i);
					// make sure they're oppisite signed 
					if((lhs * rhs) < 0) {
						values[m.indexAtLocation(i)] = 0.0;
					}
					else
						values[m.indexAtLocation(i)] = lhs + rhs;
				}
				else
					values[m.indexAtLocation(i)] = factor * values[m.indexAtLocation(i)] + m.valueAtLocation(i);
			}
	}
	
	public void timesEquals (double factor) {
		for (int i = 0; i < values.length; i++)
			values[i] *= factor;
	}

	public void elementwiseTimesEquals (ConstantMatrix m) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				values[i] *= ((DenseMatrix)m).values[i];
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] *= m.valueAtLocation(i);
	}
	
	public void elementwiseTimesEquals (ConstantMatrix m, double factor) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				values[i] *= ((DenseMatrix)m).values[i] * factor;
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] *= m.valueAtLocation(i) * factor;
	}
	
	public void divideEquals (double factor) {
		for (int i = 0; i < values.length; i++)
			values[i] /= factor;
	}

	public void elementwiseDivideEquals (ConstantMatrix m) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				values[i] /= ((DenseMatrix)m).values[i];
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] /= m.valueAtLocation(i);
	}

	public void elementwiseDivideEquals (ConstantMatrix m, double factor) {
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				values[i] /= ((DenseMatrix)m).values[i] * factor;
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				values[m.indexAtLocation(i)] /= m.valueAtLocation(i) * factor;
	}

	// xxx Perhaps make a special efficient case for binary vectors
	public double dotProduct (ConstantMatrix m) {
		double ret = 0;
		if (m instanceof DenseMatrix) {
			assert (m.singleSize() == values.length);
			for (int i = 0; i < values.length; i++)
				ret += values[i] * ((DenseMatrix)m).values[i];
		} else 
			for (int i = m.numLocations()-1; i >= 0; i--)
				ret += values[m.indexAtLocation(i)] * m.valueAtLocation(i);
		return ret;
	}
	
	public double absNorm() {
		double ret = 0;
		for (int i = 0; i < values.length; i++)
			ret += Math.abs(values[i]);
		return ret;
	}

	public double oneNorm () {
		double ret = 0;
		for (int i = 0; i < values.length; i++)
			ret += values[i];
		return ret;
	}
	
	public double twoNorm () {
		double ret = 0;
		for (int i = 0; i < values.length; i++)
			ret += values[i] * values[i];
		return Math.sqrt (ret);
	}
	
	public double infinityNorm () {
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < values.length; i++)
			if (Math.abs(values[i]) > max)
				max = Math.abs(values[i]);
		return max;
	}		

	public double oneNormalize ()	{
		double norm = oneNorm();
		for (int i = 0; i < values.length; i++)
			values[i] /= norm;
		return norm;
	}

	public double twoNormalize ()	{
		double norm = twoNorm();
		for (int i = 0; i < values.length; i++)
			values[i] /= norm;
		return norm;
	}

	public double absNormalize ()	{
		double norm = absNorm();
		if (norm > 0)
			for (int i = 0; i < values.length; i++)
				values[i] /= norm;
		return norm;
	}

	public double infinityNormalize () {
		double norm = infinityNorm();
		for (int i = 0; i < values.length; i++)
			values[i] /= norm;
		return norm;
	}
	
	public void print() {
		for (int i = 0; i < values.length; i++)
			System.out.println ("DenseMatrix["+i+"] = "+values[i]);
	}
		
	public boolean isNaN() {
		for (int i = 0; i < values.length; i++)
			if (Double.isNaN(values[i]))
				return true;
		return false;
	}


	public final void substitute (double oldValue, double newValue)
	{
		for (int i = values.length-1; i >= 0; i--)
			if (values[i] == oldValue)
				values[i] = newValue;
	}
	
	// Serialization
		
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		int i, size;
		out.writeInt (CURRENT_SERIAL_VERSION);
		if (values != null) {
			size = values.length;
			out.writeInt(size);
			for (i=0; i<size; i++) {
				out.writeDouble(values[i]);
			}
		}
		else {
			out.writeInt(NULL_INTEGER);
		}
	}
	
	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int i, size;
		int version = in.readInt ();
		size = in.readInt();
		if (size != NULL_INTEGER) {
			values = new double[size];
			for (i = 0; i<size; i++) {
				values[i] = in.readDouble();
			}
		}
		else {
			values = null;
		}
	}


	public static void plusEquals (double[] accumulator, double[] addend)
	{
		assert (accumulator.length == addend.length);
		for (int i = 0; i < addend.length; i++)
			accumulator[i] += addend[i];
	}

	public static void plusEquals (double[] accumulator, double[] addend, double factor)
	{
		assert (accumulator.length == addend.length);
		for (int i = 0; i < addend.length; i++)
			accumulator[i] += factor * addend[i];
	}
	
	public static void timesEquals (double[] accumulator, double[] product)
	{
		assert (accumulator.length == product.length);
		for (int i = 0; i < product.length; i++)
			accumulator[i] *= product[i];
	}
	
}
