package edu.umass.cs.mallet.base.types;

public final class MatrixOps
{
	public static void setAll (double[] m, double v) {
		for (int i = 0; i < m.length; i++) m[i] = v;
	}

	public static void set (double[] dest, double[] source) {
		if (source.length != dest.length)
			dest  = new double [source.length];
		System.arraycopy (source, 0, dest, 0, source.length);
	}
	
	public static void timesEquals (double[] m, double factor) {
		for (int i=0; i < m.length; i++)
			m[i] *= factor;
	}

	public static void plusEquals (double[] m, double toadd) {
		for (int i=0; i < m.length; i++)
			m[i] += toadd;
	}

	public static void plusEquals (double[] m1, double[] m2) {
		assert (m1.length == m2.length) : "unequal lengths\n";
		for (int i=0; i < m1.length; i++) {
			if (Double.isInfinite(m1[i]) && Double.isInfinite(m2[i]) && (m1[i]*m2[i] < 0))
				m1[i] = 0.0;
			else
				m1[i] += m2[i];
		}
	}
	
	public static void plusEquals (double[] m1, double[] m2,  double factor) {
		assert (m1.length == m2.length) : "unequal lengths\n";
		for (int i=0; i < m1.length; i++) {
			double m1i = m1[i];
			double m2i = m2[i];
			if (Double.isInfinite(m1i) && Double.isInfinite(m2i) && (m1[i]*m2[i] < 0)) 
			 m1[i] = 0.0;
			else  m1[i] += m2[i] * factor;
		}
	}

	public static double dot (double[] m1, double[] m2) {
		assert (m1.length == m2.length) : "m1.length != m2.length\n";
		double ret = 0.0;
		for (int i=0; i < m1.length; i++) 			
			ret += m1[i] * m2[i];
		return ret;
	}

	public static double absNorm (double[] m) {
		double ret = 0;
		for (int i = 0; i < m.length; i++)
			ret += Math.abs(m[i]);
		return ret;
	}
	
	public static double twoNorm (double[] m) {
		double ret = 0;
		for (int i = 0; i < m.length; i++)
			ret += m[i] * m[i];
		return Math.sqrt (ret);
	}

	
	public static double oneNorm (double[] m) {
		double ret = 0;
		for (int i = 0; i < m.length; i++)
			ret += m[i];
		return ret;
	}

	public static double absNormalize (double[] m) {
		double norm = absNorm(m);
		if (norm > 0)
			for (int i = 0; i < m.length; i++)
				m[i] /= norm;
		return norm;
	}

	public static double twoNormalize (double[] m) {
		double norm = twoNorm(m);
		if (norm > 0)
			for (int i = 0; i < m.length; i++)
				m[i] /= norm;
		return norm;
	}

	public static void substitute (double[] m, double oldValue, double newValue) {
		for (int i = m.length-1; i >= 0; i--)
			if (m[i] == oldValue)
				m[i] = newValue;
	}

	/** If "ifSelected" is false, it reverses the selection.  If
			"fselection" is null, this implies that all features are
			selected; all values in the row will be changed unless
			"ifSelected" is false. */
	public static final void rowSetAll (double[] m, int nc, int ri, double v, FeatureSelection fselection, boolean ifSelected) {
		if (fselection == null) {
			if (ifSelected == true) {
				for (int ci = 0; ci < nc; ci++)
					m[ri*nc+ci] = v;
			}
		} else {
			// xxx Temporary check for full selection
			//assert (fselection.nextDeselectedIndex (0) == nc);
			for (int ci = 0; ci < nc; ci++)
				if (fselection.contains(ci) ^ !ifSelected)
					m[ri*nc+ci] = v;
		}
	}
	
	public static double rowDotProduct (double[] m, int nc, int ri,
																			Vector v, int maxCi,
																			FeatureSelection selection) {
		return rowDotProduct (m, nc, ri, v, 1, maxCi, selection);
	}

	public static double rowDotProduct (double[] m, int nc, int ri,
																			Vector v, double factor, int maxCi,
																			FeatureSelection selection) {
		double ret = 0;
		if (selection != null) {
			for (int cil = 0; cil < v.numLocations(); cil++) {
				int ci = v.indexAtLocation (cil);
				if (selection.contains(ci) && ci < nc && ci <= maxCi)
					ret += m[ri*nc+ci] * v.valueAtLocation(cil) * factor;
			}
		} else {
			for (int cil = 0; cil < v.numLocations(); cil++) {
				int ci = v.indexAtLocation (cil);
				if (ci < nc && ci <= maxCi)
					ret += m[ri*nc+ci] * v.valueAtLocation(cil) * factor;
			}
		}
		return ret;
	}

	public static final void rowPlusEquals (double[] m, int nc, int ri,
																					Vector v, double factor) {
		for (int vli = 0; vli < v.numLocations(); vli++) 
			m[ri*nc+v.indexAtLocation(vli)] += v.valueAtLocation(vli) * factor;

	}

	public static boolean isNaN(double[] m) {
		for (int i = 0; i < m.length; i++)
			if (Double.isNaN(m[i]))
				return true;
		return false;
	}

	public static final void print (double[] m) {
		for (int i = 0; i < m.length; i++) {
				System.out.print (" " + m[i]);
		}
		System.out.println("");
	}
	
}
