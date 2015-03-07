/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/**
	 A vector that only stores non-zero values.

	 Instances of this class are nearly immutable.  They can be changed to make them binary
	 or non-binary.  Also subclass AugmentableSparseVector is mutable by adding new values.
	 
   @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
*/

package edu.umass.cs.mallet.base.types;

import java.util.Arrays;
import edu.umass.cs.mallet.base.util.PropertyList;
import java.io.*;

public class ConstantVector implements ConstantMatrix, Vector, Serializable
{
	int[] indices;												// if this is null, then the vector is dense
	double[] values;											// if this is null, then the vector is binary

	
	/** If "indices" is null, the vector will be dense.  If "values" is
			null, the vector will be binary.  The capacity and size arguments are
			used by AugmentableFeatureVector. */
	protected ConstantVector (int[] indices, double[] values, 
														int capacity, int size,
														boolean copy,
														boolean checkIndicesSorted,
														boolean removeDuplicates)
	{
		// "size" was pretty much ignored??? Why?
		int length;
		//if (indices == null)
		//length = values.length;
		//else {
		//length = indices.length;
		//for (int i = 0; i < indices.length; i++)
		//assert (indices[i] >= 0);
		//}
		length = size;
		if (capacity < length)
			capacity = length;
		assert (size <= length);
		assert (values == null || indices == null || indices.length == values.length);
		if (copy || capacity > length) {
			if (indices == null)
				this.indices = null;
			else {
				this.indices = new int[capacity];
				System.arraycopy (indices, 0, this.indices, 0, length);
			}
			if (values == null)
				this.values = null;
			else {
				this.values = new double[capacity];
				System.arraycopy (values, 0, this.values, 0, length);
			}
		} else {
			this.indices = indices;
			this.values = values;
		}
		if (checkIndicesSorted)
			sortIndices ();										// This also removes duplicates
		else if (removeDuplicates)
			removeDuplicates (0);
	}


	// Create a dense Vector
	public ConstantVector (double[] values, boolean copy)	{
		this (null, values, values.length, values.length, copy, false, false);
	}

	public ConstantVector (double[] values) { this (values, true); }

	public ConstantVector (int size, double fillValue) {
		this (newArrayOfValue (size, fillValue), false); }

	
	public ConstantVector (int[] indices, double[] values, 
												 boolean copy, boolean checkIndicesSorted,
												 boolean removeDuplicates)
	{
		this (indices, values, indices.length, indices.length,
					copy, checkIndicesSorted, removeDuplicates);
	}

	public ConstantVector (int[] indices, double[] values) {
		this (indices, values, true, true, true); }
	public ConstantVector (int[] indices, double[] values, boolean copy) {
		this (indices, values, copy, true, true); }
	public ConstantVector (int[] indices, double[] values, boolean copy,
												 boolean checkIndicesSorted) {
		this (indices, values, copy, checkIndicesSorted, true); }


	// Create a vector that is possibly binary or non-binary
	public ConstantVector (int[] indices,
												 boolean copy,
												 boolean checkIndicesSorted,
												 boolean removeDuplicates,
												 boolean binary)
	{
		this (indices, binary ? null : newArrayOfValue(indices.length,1.0), indices.length, indices.length,
					copy, checkIndicesSorted, removeDuplicates);
	}

	// Create a binary vector
	public ConstantVector (int[] indices,
												 int capacity, int size,
												 boolean copy,
												 boolean checkIndicesSorted,
												 boolean removeDuplicates)
	{
		this (indices, null, capacity, size, copy, checkIndicesSorted, removeDuplicates);
	}

	public ConstantVector (int[] indices, boolean copy, boolean checkIndicesSorted) {
		this (indices, null, copy, checkIndicesSorted, true);	}
	public ConstantVector (int[] indices, boolean copy) {
		this (indices, null, copy, true, true); }
	public ConstantVector (int[] indices) {
		this (indices, null, true, true, true); }
	/** An empty vector, with all zero values */
	public ConstantVector () {
		this (new int[0], new double[0], false, false); }

	public ConstantVector (Alphabet dict, PropertyList pl, boolean binary,
												 boolean growAlphabet)
	{
		if (pl == null) {
			// xxx Fix ConstantVector so that it can properly represent a vector that has all zeros.
			// Does this work?
			indices = new int[0];
			values = null;
			return;
		}
		AugmentableFeatureVector afv = new AugmentableFeatureVector (dict, binary);
		//afv.print();
		//System.out.println ("ConstantVector binary="+binary);
		//pl.print();
		PropertyList.Iterator iter = pl.numericIterator();
		while (iter.hasNext()) {
			iter.nextProperty();
			//System.out.println ("ConstantVector adding "+iter.getKey()+" "+iter.getNumericValue());
			int index = dict.lookupIndex(iter.getKey(), growAlphabet);
			if (index >=0) {
				afv.add (index, iter.getNumericValue());
			}
			//System.out.println ("ConstantVector afv adding "+iter.getKey()+" afv.numLocations="+afv.numLocations());
		}
		//afv.print();
		// xxx Not so efficient?
		ConstantVector sv = afv.toConstantVector();
		//System.out.println ("ConstantVector cv.numLocations="+sv.numLocations());
		this.indices = sv.indices;
		this.values = sv.values;
	}
	public ConstantVector (Alphabet dict, PropertyList pl, boolean binary)
	{
		this(dict, pl, binary, true);
	}

	private static double[] newArrayOfValue (int length, double value)
	{
		double[] ret = new double[length];
		Arrays.fill (ret, value);
		return ret;
	}


	
	public ConstantVector toConstantVector () {
		return this;
	}
	
	public boolean isBinary () { return values == null; }
	public void makeBinary () { throw new UnsupportedOperationException ("Not yet implemented"); }
	public void makeNonBinary () { throw new UnsupportedOperationException ("Not yet implemented"); }
	
	public int getNumDimensions () { return 1; }
	// xxx What do we return for the length?  It could be higher than this index.
	public int getDimensions (int[] sizes) {
		if (indices == null)
			sizes[0] = values.length;
		else
			// xxx This is pretty unsatisfactory, since there may be zero
			// values above this location.
			sizes[0] = indices[indices.length-1];
		return 1;
	}

	// xxx This is just the number of non-zero entries...
	// This is different behavior than Matrix2!!
	public int numLocations () {
		return (values == null
						? (indices == null
							 ? 0
							 : indices.length)
						: values.length);
	}

	public int location (int index) {
		if (indices == null)
			return index;
		else
			return Arrays.binarySearch (indices, index);
	}

	public double valueAtLocation (int location) { return values == null ? 1.0 : values[location]; }
	public int indexAtLocation (int location) { return indices == null ? location : indices[location]; }
	
	public double value (int[] indices) {
		assert (indices.length == 1);
		if (indices == null)
			return values[indices[0]];
		else
			return values[location(indices[0])];
	}
		
	public double value (int index) {
		if (indices == null)
			return values[index];
		else {
			int loc = location(index);
			if (loc < 0)
				return 0.0;
			else if (values == null)
				return 1.0;
			else
				return values[location(index)];
		}
	}

	public void addTo (double[] accumulator, double scale)
	{
		if (indices == null) {
			for (int i = 0; i < values.length; i++)
				accumulator[i] += values[i] * scale;
		} else if (values == null) {
			for (int i = 0; i < indices.length; i++)
				accumulator[indices[i]] += scale;
		} else {
			for (int i = 0; i < indices.length; i++)
				accumulator[indices[i]] += values[i] * scale;
		}
	}

	public void addTo (double[] accumulator) {
		addTo (accumulator, 1.0);
	}
	
	public ConstantMatrix cloneMatrix () {
		if (indices == null)
			return new ConstantVector (values);
		else
			return new ConstantVector (indices, values, true, false, false);
	}

	public ConstantMatrix cloneMatrixZeroed () {
		if (indices == null)
			return new ConstantVector (new double[values.length]);
		else {
			int[] newIndices = new int[indices.length];
			System.arraycopy (indices, 0, newIndices, 0, indices.length);
			return new ConstantVector (newIndices, new double[values.length], true, false, false);
		}
	}
	
	public int singleIndex (int[] indices) { assert (indices.length == 1); return indices[0]; }
	public void singleToIndices (int i, int[] indices) { indices[0] = i; }
	public double singleValue (int i) { return value(i); }
	public int singleSize () {
		if (indices == null)
			return values.length;
		else if (indices.length == 0)
			return 0;
		else
			// This is just the highest index that will have non-zero value.
			// The full size of this dimension is "unknown"
			return indices[indices.length-1];
	}

	// xxx Perhaps remove?
	private void setValue (int[] indices, double value) {
		assert (indices.length == 1);
		if (indices == null)
			values[indices[0]] = value;
		else {
			int loc = location(indices[0]);
			if (loc < 0)
				throw new UnsupportedOperationException ("Can't yet insert values into a sparse Vector.");
			else
				values[loc] = value;
		}
	}

	// xxx Perhaps remove?
	private void setValue (int index, double value) {
		if (indices == null)
			values[index] = value;
		else {
			int loc = location(index);
			if (loc < 0)
				throw new UnsupportedOperationException ("Can't yet insert values into a sparse Vector.");
			else
				values[loc] = value;
		}
	}

	public double dotProduct (ConstantMatrix m) {
		if (m instanceof ConstantVector) return dotProduct ((ConstantVector)m);
		else if (m instanceof DenseVector) return dotProduct ((DenseVector)m);
		else throw new IllegalArgumentException ("Unrecognized Matrix type "+m.getClass());
	}

	public double dotProduct (DenseVector v) {
		double ret = 0;
		if (values == null)
			for (int i = 0; i < indices.length; i++)
				ret += v.value(indices[i]);
		else
			for (int i = 0; i < indices.length; i++)
				ret += values[i] * v.value(indices[i]);
		return ret;
	}

	// added by dmetzler
	public double dotProduct (ConstantVector v) {
		/*double ret = 0.0;
			for(int i = 0; i < indices.length; i++) {
			if(values == null)
			ret += v.value(indices[i]);
			else
			ret += values[i]*v.value(indices[i]);
			}
			return ret;*/
		double ret = 0.0;
		ConstantVector vShort = null;
		ConstantVector vLong = null;
		// this ensures minimal computational effort
		if(indices.length > v.getIndices().length) {
			vShort = v;
			vLong = this;
		}
		else {
			vShort = this;
			vLong = v;
		}
		double [] valuesShort = vShort.getValues();
		int [] indicesShort = vShort.getIndices();
		for(int i = 0; i < valuesShort.length; i++) {
			if(values == null)
				ret += vLong.value(indicesShort[i]);
			else
				ret += valuesShort[i]*vLong.value(indicesShort[i]);
		}
		return ret;
	}
	
	// added by dmetzler
	public ConstantVector vectorAdd(ConstantVector v, double scale) {
		if(indices != null) { // sparse ConstantVector
			int [] ind = v.getIndices();
			double [] val = v.getValues();
			int [] newIndices = new int[ind.length+indices.length];
			double [] newVals = new double[ind.length+indices.length];
			for(int i = 0; i < indices.length; i++) {
		    newIndices[i] = indices[i];
		    newVals[i] = values[i];
			}
			for(int i = 0; i < ind.length; i++) {
		    newIndices[i+indices.length] = ind[i];
		    newVals[i+indices.length] = scale*val[i];
			}
			return new ConstantVector(newIndices, newVals, true, true, false);
		}
		int [] newIndices = new int[values.length];
		double [] newVals = new double[values.length]; // dense ConstantVector
		int curPos = 0;
		for(int i = 0; i < values.length; i++) {
			double val = values[i]+scale*v.value(i);
			if(val != 0.0) {
		    newIndices[curPos] = i;
		    newVals[curPos++] = val;
			}
		}
		return new ConstantVector(newIndices, newVals, true, true, false);
	}

	// accessor methods added by dmetzler
	// necessary for the SVM implementation!
	public int [] getIndices() {
		return indices;
	}
	
	public double [] getValues() {
		return values;
	}
	
	public double oneNorm () {
		double ret = 0;
		if (values == null)
			return indices.length;
		for (int i = 0; i < values.length; i++)
			ret += values[i];
		return ret;
	}

	public double absNorm () {
		double ret = 0;
		if (values == null)
			return indices.length;
		for (int i = 0; i < values.length; i++)
			ret += Math.abs(values[i]);
		return ret;
	}
	
	public double twoNorm () {
		double ret = 0;
		if (values == null)
			return Math.sqrt (indices.length);
		for (int i = 0; i < values.length; i++)
			ret += values[i] * values[i];
		return Math.sqrt (ret);
	}
	
	public double infinityNorm () {
		if (values == null)
			return 1.0;
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < values.length; i++)
			if (Math.abs(values[i]) > max)
				max = Math.abs(values[i]);
		return max;
	}		

	public void print() {
		for (int i = 0; i < values.length; i++)
			System.out.println ("ConstantVector["+indices[i]+"] = "+(values==null ? 1.0 : values[i]));
	}
		
	public boolean isNaN() {
		if (values == null)
			return false;
		for (int i = 0; i < values.length; i++)
			if (Double.isNaN(values[i]))
				return true;
		return false;
	}

	
	protected void sortIndices ()
	{
		int numDuplicates = 0;
		if (indices == null)
			// It's dense, and thus by definition sorted.
			return;
		if (values == null)
			java.util.Arrays.sort (indices);
		else {
			// Just BubbleSort; this is efficient when already mostly sorted.
			// Note that we BubbleSort from the the end forward; this is most efficient
			//  when we have added a few additional items to the end of a previously sorted list.
			//  We could be much smarter if we remembered the highest index that was already sorted
			for (int i = indices.length-1; i >= 0; i--) {
				boolean swapped = false;
				for (int j = 0; j < i; j++)
					if (indices[j] == indices[j+1] && i == indices.length-1) {
						numDuplicates++;
					} else if (indices[j] > indices[j+1]) {
						// Swap both indices and values
						int f;
						f = indices[j];
						indices[j] = indices[j+1];
						indices[j+1] = f;
						if (values != null) {
							double v;
							v = values[j];
							values[j] = values[j+1];
							values[j+1] = v;
						}
						swapped = true;
					}
				if (!swapped)
					break;
			}
		}

		//if (values == null)
		numDuplicates = 0;
		for (int i = 1; i < indices.length; i++)
		    if (indices[i-1] == indices[i])
			numDuplicates++;

		if (numDuplicates > 0)
			removeDuplicates (numDuplicates);
	}

	// Argument zero is special value meaning that this function should count them.
	protected void removeDuplicates (int numDuplicates)
	{
		if (numDuplicates == 0)
			for (int i = 1; i < indices.length; i++)
				if (indices[i-1] == indices[i])
					numDuplicates++;
		if (numDuplicates == 0)
			return;
		int[] newIndices = new int[indices.length - numDuplicates];
		double[] newValues = values == null ? null : new double[indices.length - numDuplicates];
		newIndices[0] = indices[0];
		if (values != null) newValues[0] = values[0];
		for (int i = 1, j = 1; i < indices.length; i++) {
			if (indices[i] == indices[i-1]) {
				if (newValues != null)
					newValues[j-1] += values[i];
			} else {
				newIndices[j] = indices[i];
				if (values != null)
					newValues[j] = values[i];
				j++;
			}
		}
		this.indices = newIndices;
		this.values = newValues;
	}


    /** Returns a <CODE>ConstantVector</CODE> with the same Alphabet and whose 
	entries are the expected values of the feature values of the 
	<CODE>InstanceList</CODE>. This implies the returned feature vector 
	will not have binary values. */
        public static ConstantVector mean( InstanceList instances )
        {

	    if (instances==null || instances.size()==0)
		return null;
	    
	    int dataSz = ((ConstantVector)(instances.getInstance(0).getData())).numLocations();
	    
	    
	    double[] dataMean = new double[dataSz];

	    InstanceList.Iterator instanceItr;
	    Instance instance;
	    ConstantVector cv;

	    for (int i=0 ; i<dataSz ; i++ ) 
	    {

		instanceItr = instances.iterator();

		while (instanceItr.hasNext())
		{    
		    instance = (Instance)instanceItr.next();
		    cv = (ConstantVector)(instance.getData());

		    if (cv.numLocations()!=dataSz)
			throw new IllegalArgumentException("Instances are not of the same dimension.");
		    
		    dataMean[i] += cv.valueAtLocation(i);
		}

		// Summing without normalizing first COULD cause overflow 
		// problems, but we're banking on the savings by only normalizing
		// once
		dataMean[i] /= instances.size();
	    }
	    
	    return new ConstantVector( dataMean );

	}

    /** Returns a <CODE>ConstantVector</CODE> with the same 
	<CODE>Alphabet</CODE> and whose entries are the variance of the 
	feature values of the instance list. This implies the returned feature 
	vector will not have binary values.
	
	@param unbiased Normalizes by N-1 when true, and by N otherwise.
    */
        public static ConstantVector variance( InstanceList instances, boolean unbiased )
        {

	    if (instances==null || instances.size()==0)
		return null;
	    
	    double[] dataMean = mean(instances).values;

	    double[] dataVar = new double[dataMean.length];

	    InstanceList.Iterator instanceItr;
	    Instance instance;
	    ConstantVector cv;

	    for (int i=0 ; i<dataMean.length ; i++ ) 
	    {

		instanceItr = instances.iterator();

		while (instanceItr.hasNext())
		{    
		    instance = (Instance)instanceItr.next();
		    cv = (ConstantVector)(instance.getData());

		    if (cv.numLocations()!=dataMean.length)
			throw new IllegalArgumentException("Instances are not of the same dimension.");
		    

		    dataVar[i] +=  cv.valueAtLocation(i)*cv.valueAtLocation(i);
		}
		
		// var = (x^2 - n*mu^2)/(n-1)
		dataVar[i] = ( dataVar[i] - 
			       instances.size()*dataMean[i]*dataMean[i] ) /
		    ( unbiased ? instances.size() - 1 : instances.size() );
	    }

	    return new ConstantVector( dataVar );
	}
			          
	    /** Returns unbiased variance */
        public static ConstantVector variance( InstanceList instances )
        {

	    return variance( instances, true );

	}

	/** Square root of variance.
	    
	@param unbiased Normalizes variance by N-1 when true, and by N otherwise.
	@see variance */
        public static ConstantVector stdev( InstanceList instances, 
					    boolean unbiased )
        {
	    
	    if (instances.size()==0)
		return null;
	    
	    double[] dataVar = variance(instances, unbiased).values;
	    double[] dataStd = new double[dataVar.length];
	    
	    for (int i=0 ; i<dataVar.length ; i++ ) 
		
		dataStd[i] = Math.sqrt( dataVar[i] );

	    
	    return new ConstantVector( dataStd );
	}

        /** Square root of unbiased variance. */

	public static ConstantVector stdev( InstanceList instances )
	{

	    return stdev( instances, true );

	}

	// Serialization 
		
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeInt (indices == null ? -1 : indices.length);
		out.writeInt (values == null ? -1 : values.length);
		if (indices != null)
			for (int i = 0; i < indices.length; i++)
				out.writeInt (indices[i]);
		if (values != null)
			for (int i = 0; i < values.length; i++)
				out.writeDouble (values[i]);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		int indicesSize = in.readInt();
		int valuesSize = in.readInt();
		if (indicesSize >= 0) {
			indices = new int[indicesSize];
			for (int i = 0; i < indicesSize; i++)
				indices[i] = in.readInt();
		}
		if (valuesSize >= 0) {
			values = new double[indicesSize];
			for (int i = 0; i < valuesSize; i++)
				values[i] = in.readDouble();
		}
	}
}
