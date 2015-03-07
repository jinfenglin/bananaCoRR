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

// Analogous to base.types.classify.Classifier

import java.util.Iterator;
import java.util.ArrayList;
import java.util.logging.*;
import edu.umass.cs.mallet.base.pipe.Pipe;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.InstanceList;
import edu.umass.cs.mallet.base.types.Instance;
import edu.umass.cs.mallet.base.types.Sequence;
import edu.umass.cs.mallet.base.types.ArraySequence;
import edu.umass.cs.mallet.base.types.SequencePair;
import edu.umass.cs.mallet.base.types.SequencePairAlignment;
import edu.umass.cs.mallet.base.types.Label;
import edu.umass.cs.mallet.base.types.LabelAlphabet;
import edu.umass.cs.mallet.base.types.LabelVector;
import edu.umass.cs.mallet.base.types.DenseVector;
import edu.umass.cs.mallet.base.util.MalletLogger;
import java.io.*;

// Variable name key:
// "ip" = "input position"
// "op" = "output position"

public abstract class Transducer implements Serializable
{
	private static Logger logger = MalletLogger.getLogger(Transducer.class.getName());

	{
		// xxx Why isn't this resulting in printing the log messages?
		//logger.setLevel (Level.FINE);
		//logger.addHandler (new StreamHandler (System.out, new SimpleFormatter ()));
		//System.out.println ("Setting level to finer");
		//System.out.println ("level = " + logger.getLevel());
		//logger.warning ("Foooooo");
	}
	
	public static final double ZERO_COST = 0;
	public static final double INFINITE_COST = Double.POSITIVE_INFINITY;

	
	//private Stack availableTransitionIterators = new Stack ();

	// Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 1;
	private static final int NO_PIPE_VERSION = 0;


	private void writeObject (ObjectOutputStream out) throws IOException {
		int i, size;
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeObject(inputPipe);
		out.writeObject(outputPipe);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int size, i;
		int version = in.readInt ();
		if (version == NO_PIPE_VERSION) {
			inputPipe = null;
			outputPipe = null;
		}
		else {
			inputPipe = (Pipe) in.readObject();
			outputPipe = (Pipe) in.readObject();
		}
	}
	
	public abstract static class State implements Serializable
	{
		double initialCost = 0;
		double finalCost = 0;

		public abstract String getName();
		public abstract int getIndex ();
		public double getInitialCost () { return initialCost; }
		public void setInitialCost (double c) { initialCost = c; }
		public double getFinalCost () { return finalCost; }
		public void setFinalCost (double c) { finalCost = c; }
		//public Transducer getTransducer () { return (Transducer)this; }
		//public abstract TransitionIterator transitionIterator (Object input);

		// Pass negative positions for a sequence to request "epsilon
		// transitions" for either input or output.  (-position-1) should be
		// the position in the sequence after which we are trying to insert
		// the espilon transition.
		public abstract TransitionIterator transitionIterator
		(Sequence input,	int inputPosition, Sequence output, int outputPosition);

		/*
		public abstract TransitionIterator transitionIterator {
			if (availableTransitionIterators.size() > 0)
				return ((TransitionIterator)availableTransitionIterators.pop()).initialize
					(State source, Sequence input,	int inputPosition, Sequence output, int outputPosition);
			else
				return newTransitionIterator
					(Sequence input,	int inputPosition, Sequence output, int outputPosition);
		}
		*/

		// Pass negative input position for a sequence to request "epsilon
		// transitions".  (-position-1) should be the position in the
		// sequence after which we are trying to insert the espilon
		// transition.
		public TransitionIterator transitionIterator (Sequence input,
																									int inputPosition) {
			return transitionIterator (input, inputPosition, null, 0);
		}

		// For generative transducers:
		// Return all possible transitions, independent of input
		public TransitionIterator transitionIterator () {
			return transitionIterator (null, 0, null, 0);
		}

		// For trainable transducers:
		public void incrementInitialCount (double count) {
			throw new UnsupportedOperationException (); }
		public void incrementFinalCount (double count) {
			throw new UnsupportedOperationException (); }
		

		// Serialization
		
		private static final long serialVersionUID = 1;
		private static final int CURRENT_SERIAL_VERSION = 0;


		private void writeObject (ObjectOutputStream out) throws IOException {
			int i, size;
			out.writeInt (CURRENT_SERIAL_VERSION);
			out.writeDouble(initialCost);
			out.writeDouble(finalCost);
		}
	

		private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
			int size, i;
			int version = in.readInt ();
			initialCost = in.readDouble();
			finalCost = in.readDouble();
		}
	}

	public abstract static class TransitionIterator implements Iterator, Serializable
	{
		//public abstract void initialize (Sequence input, int inputPosition,
		//Sequence output, int outputPosition);
		public abstract boolean hasNext ();
		public abstract State nextState ();	// returns the destination state
		public Object next () { return nextState(); }
		public void remove () {
			throw new UnsupportedOperationException (); }
		public abstract Object getInput ();
		public abstract Object getOutput ();
		public abstract double getCost ();
		public abstract State getSourceState ();
		public abstract State getDestinationState ();
		// In future these will allow for transition that consume variable amounts of the sequences
		public int getInputPositionIncrement () { return 1; }
		public int getOutputPositionIncrement () { return 1; }
		//public abstract Transducer getTransducer () {return getSourceState().getTransducer();}
		// For trainable transducers:
		public void incrementCount (double count) {
			throw new UnsupportedOperationException (); }

		// Serialization

		private static final long serialVersionUID = 1;
		private static final int CURRENT_SERIAL_VERSION = 0;
		
		
		private void writeObject (ObjectOutputStream out) throws IOException {
			int i, size;
			out.writeInt (CURRENT_SERIAL_VERSION);
		}
		
		private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
			int size, i;
			int version = in.readInt ();
		}
		
	}

	/** A pipe that should produce a Sequence in the "data" slot, (and possibly one in the "target" slot also */
	protected Pipe inputPipe;

	/** A pipe that should expect a ViterbiPath in the "target" slot,
			and should produce something printable in the "source" slot that
			indicates the results of transduction. */
	protected Pipe outputPipe;

	public Pipe getInputPipe () { return inputPipe; }
	public Pipe getOutuptPipe () { return outputPipe; }

	
	/** We aren't really a Pipe subclass, but this method works like Pipes' do. */
	public Instance pipe (Instance carrier)
	{
		carrier.setTarget(viterbiPath ((Sequence)carrier.getData()));
		return carrier;
	}

	// xxx Enrich this later.
	// Perhaps to something like:
	// public Transduction transduce (Instance instance)
	// public Transduction transduce (Object obj)
	public Instance transduce (Instance instance)
	{
		throw new UnsupportedOperationException ("Not yet implemented");
	}

	public abstract int numStates ();
	public abstract State getState (int index);
	
	// Note that this method is allowed to return states with infinite initialCost.
	public abstract Iterator initialStateIterator ();

	// Some transducers are "generative", meaning that you can get a
	// sequence out of them without giving them an input sequence.  In
	// this case State.transitionIterator() should return all available
	// transitions, but attempts to obtain the input and cost fields may
	// throw an exception.
	// xxx Why could obtaining "cost" be a problem???
	public boolean canIterateAllTransitions () { return false; }

	// If true, this is a "generative transducer".  In this case
	// State.transitionIterator() should return transitions that have
	// valid input and cost fields.  True returned here should imply
	// that canIterateAllTransitions() is true.
	public boolean isGenerative () { return false; }

	public boolean isTrainable () { return false; }
	// If f is true, and it was already trainable, this has same effect as reset()
	public void setTrainable (boolean f) {
		if (f) throw new IllegalStateException ("Cannot be trainable."); }
	public boolean train (InstanceList instances) {
		throw new UnsupportedOperationException ("Not trainable."); }

	public double averageTokenAccuracy (InstanceList ilist)
	{
		double accuracy = 0;
		for (int i = 0; i < ilist.size(); i++) {
			Instance instance = ilist.getInstance(i);
			Sequence input = (Sequence) instance.getData();
			Sequence output = (Sequence) instance.getTarget();
			assert (input.size() == output.size());
			double pathAccuracy = viterbiPath(input).tokenAccuracy(output);
			accuracy += pathAccuracy;
			logger.info ("Transducer path accuracy = "+pathAccuracy);
		}
		return accuracy/ilist.size();
	}

	public double averageTokenAccuracy (InstanceList ilist, String fileName)
	{
		double accuracy = 0;
		PrintWriter out;
		File f = new File(fileName);
		try {
			out = new PrintWriter(new FileWriter(f));
		}
		catch (IOException e) {
			out = null;
		}
		for (int i = 0; i < ilist.size(); i++) {
			Instance instance = ilist.getInstance(i);
			Sequence input = (Sequence) instance.getData();
			Sequence output = (Sequence) instance.getTarget();
			assert (input.size() == output.size());
			double pathAccuracy = viterbiPath(input).tokenAccuracy(output, out);
			accuracy += pathAccuracy;
			logger.info ("Transducer path accuracy = "+pathAccuracy);
		}
		out.close();
		return accuracy/ilist.size();
	}

	// Treat the costs as if they are -log(probabilies); we will
	// normalize them if necessary
	public SequencePairAlignment generatePath ()
	{
		if (isGenerative() == false)
			throw new IllegalStateException ("Transducer is not generative.");
		ArrayList initialStates = new ArrayList ();
		Iterator iter = initialStateIterator ();
		while (iter.hasNext()) { initialStates.add (iter.next()); }
		// xxx Not yet finished.
		throw new UnsupportedOperationException ();
	}


	
	public Lattice forwardBackward (Sequence inputSequence)
	{
		return forwardBackward (inputSequence, null, false);
	}

	public Lattice forwardBackward (Sequence inputSequence, boolean increment)
	{
		return forwardBackward (inputSequence, null, increment);
	}
	
	public Lattice forwardBackward (Sequence inputSequence, Sequence outputSequence)
	{
		return forwardBackward (inputSequence, outputSequence, false);
	}

	public Lattice forwardBackward (Sequence inputSequence, Sequence outputSequence, boolean increment)
	{
		return forwardBackward (inputSequence, outputSequence, increment, null);
	}
	public Lattice forwardBackward (Sequence inputSequence, Sequence outputSequence, boolean increment,
																	LabelAlphabet outputAlphabet)
	{
		// xxx We don't do epsilon transitions for now
		assert (outputSequence == null
						|| inputSequence.size() == outputSequence.size());
		return new Lattice (inputSequence, outputSequence, increment, outputAlphabet);
	}

	// Remove this method?
	// If "increment" is true, call incrementInitialCount, incrementFinalCount and incrementCount
	private Lattice forwardBackward (SequencePair inputOutputPair, boolean increment) {
		return this.forwardBackward (inputOutputPair.input(), inputOutputPair.output(), increment);
	}
	
	// xxx Include methods like this?
	// ...making random selections proportional to cost
	//public Transduction transduce (Object[] inputSequence)
	//{	throw new UnsupportedOperationException (); }
	//public Transduction transduce (Sequence inputSequence)
	//{	throw new UnsupportedOperationException (); }


	public class Lattice // ?? extends SequencePairAlignment, but there isn't just a single output!
	{
		// "ip" == "input position", "op" == "output position", "i" == "state index"
		double cost;
		Sequence input, output;
		LatticeNode[][] nodes;			 // indexed by ip,i
		int latticeLength;
		// xxx Now that we are incrementing here directly, there isn't
		// necessarily a need to save all these arrays...
		// -log(probability) of being in state "i" at input position "ip"
		double[][] gammas;					 // indexed by ip,i
		LabelVector labelings[];			 // indexed by op, created only if "outputAlphabet" is non-null in constructor

		private LatticeNode getLatticeNode (int ip, int stateIndex)
		{
			if (nodes[ip][stateIndex] == null)
				nodes[ip][stateIndex] = new LatticeNode (ip, getState (stateIndex));
			return nodes[ip][stateIndex];
		}

		// You may pass null for output, meaning that the lattice
		// is not constrained to match the output
		protected Lattice (Sequence input, Sequence output, boolean increment)
		{
			this (input, output, increment, null);
		}

		// If outputAlphabet is non-null, this will create a LabelVector
		// for each position in the output sequence indicating the
		// probability distribution over possible outputs at that time
		// index
		protected Lattice (Sequence input, Sequence output, boolean increment, LabelAlphabet outputAlphabet)
		{
			if (false && logger.isLoggable (Level.FINE)) {
				logger.fine ("Starting Lattice");
				logger.fine ("Input: ");
				for (int ip = 0; ip < input.size(); ip++)
					logger.fine (" " + input.get(ip));
				logger.fine ("\nOutput: ");
				if (output == null)
					logger.fine ("null");
				else
					for (int op = 0; op < output.size(); op++)
						logger.fine (" " + output.get(op));
				logger.fine ("\n");
			}

			// Initialize some structures
			this.input = input;
			this.output = output;
			// xxx Not very efficient when the lattice is actually sparse,
			// especially when the number of states is large and the
			// sequence is long.
			latticeLength = input.size()+1;
			int numStates = numStates();
			nodes = new LatticeNode[latticeLength][numStates];
			// xxx Yipes, this could get big; something sparse might be better?
			gammas = new double[latticeLength][numStates];
			// xxx Move this to an ivar, so we can save it?  But for what?
			double xis[][][] = new double[latticeLength][numStates][numStates];
			double outputCounts[][] = null;
			if (outputAlphabet != null)
				outputCounts = new double[latticeLength][outputAlphabet.size()];

			for (int i = 0; i < numStates; i++) {
				for (int ip = 0; ip < latticeLength; ip++)
					gammas[ip][i] = INFINITE_COST;
				for (int j = 0; j < numStates; j++)
					for (int ip = 0; ip < latticeLength; ip++)
						xis[ip][i][j] = INFINITE_COST;
			}

			// Forward pass
			logger.fine ("Starting Foward pass");
			boolean atLeastOneInitialState = false;
			for (int i = 0; i < numStates; i++) {
				double initialCost = getState(i).initialCost;
				//System.out.println ("Forward pass initialCost = "+initialCost);
				if (initialCost < INFINITE_COST) {
					getLatticeNode(0, i).alpha = initialCost;
					//System.out.println ("nodes[0][i].alpha="+nodes[0][i].alpha);
					atLeastOneInitialState = true;
				}
			}
			if (atLeastOneInitialState == false)
				logger.warning ("There are no starting states!");
			
			for (int ip = 0; ip < latticeLength-1; ip++)
				for (int i = 0; i < numStates; i++) {
					if (nodes[ip][i] == null || nodes[ip][i].alpha == INFINITE_COST)
						// xxx if we end up doing this a lot,
						// we could save a list of the non-null ones
						continue;
					State s = getState(i);
					TransitionIterator iter = s.transitionIterator (input, ip, output, ip);
					if (logger.isLoggable (Level.FINE))
						logger.fine (" Starting Foward transition iteration from state "
												 + s.getName() + " on input " + input.get(ip).toString()
												 + " and output "
												 + (output==null ? "(null)" : output.get(ip).toString()));
					while (iter.hasNext()) {
						State destination = iter.nextState();
						if (logger.isLoggable (Level.FINE))
							logger.fine ("Forward Lattice[inputPos="+ip
													 +"][source="+s.getName()
													 +"][dest="+destination.getName()+"]");
						LatticeNode destinationNode = getLatticeNode (ip+1, destination.getIndex());
						destinationNode.output = iter.getOutput();
						double transitionCost = iter.getCost();
						if (logger.isLoggable (Level.FINE))
							logger.fine ("transitionCost="+transitionCost
													 +" nodes["+ip+"]["+i+"].alpha="+nodes[ip][i].alpha
													 +" destinationNode.alpha="+destinationNode.alpha);
						destinationNode.alpha = sumNegLogProb (destinationNode.alpha,
																									 nodes[ip][i].alpha + transitionCost);
						//System.out.println ("destinationNode.alpha <- "+destinationNode.alpha);
					}
				}

			// Calculate total cost of Lattice.  This is the normalizer
			cost = INFINITE_COST;
			for (int i = 0; i < numStates; i++)
				if (nodes[latticeLength-1][i] != null) {
					// Note: actually we could sum at any ip index,
					// the choice of latticeLength-1 is arbitrary
					//System.out.println ("Ending alpha, state["+i+"] = "+nodes[latticeLength-1][i].alpha);
					//System.out.println ("Ending beta,  state["+i+"] = "+getState(i).finalCost);
					cost = sumNegLogProb (cost,
																(nodes[latticeLength-1][i].alpha + getState(i).finalCost));
				}
			// Cost is now an "unnormalized cost" of the entire Lattice
			//assert (cost >= 0) : "cost = "+cost;

			// If the sequence has infinite cost, just return.
			// Usefully this avoids calling any incrementX methods.
			// It also relies on the fact that the gammas[][] and .alpha and .beta values
			// are already initialized to values that reflect infinite cost
			// xxx Although perhaps not all (alphas,betas) exactly correctly reflecting?
			if (cost == INFINITE_COST)
				return;

			// Backward pass
			for (int i = 0; i < numStates; i++)
				if (nodes[latticeLength-1][i] != null) {
					State s = getState(i);
					nodes[latticeLength-1][i].beta = s.finalCost;
					gammas[latticeLength-1][i] =
						nodes[latticeLength-1][i].alpha + nodes[latticeLength-1][i].beta - cost;
					if (increment) {
						double p = Math.exp(-gammas[latticeLength-1][i]);
						assert (p < INFINITE_COST && !Double.isNaN(p))
							: "p="+p+" gamma="+gammas[latticeLength-1][i];
						s.incrementFinalCount (p);
					}
				}

			for (int ip = latticeLength-2; ip >= 0; ip--) {
				for (int i = 0; i < numStates; i++) {
					if (nodes[ip][i] == null || nodes[ip][i].alpha == INFINITE_COST)
						// Note that skipping here based on alpha means that beta values won't
						// be correct, but since alpha is infinite anyway, it shouldn't matter.
						continue;
					State s = getState(i);
					TransitionIterator iter = s.transitionIterator (input, ip, output, ip);
					while (iter.hasNext()) {
						State destination = iter.nextState();
						if (logger.isLoggable (Level.FINE))
							logger.fine ("Backward Lattice[inputPos="+ip
													 +"][source="+s.getName()
													 +"][dest="+destination.getName()+"]");
						int j = destination.getIndex();
						LatticeNode destinationNode = nodes[ip+1][j];
						if (destinationNode != null) {
							double transitionCost = iter.getCost();
							assert (!Double.isNaN(transitionCost));
							//							assert (transitionCost >= 0);  Not necessarily
							double oldBeta = nodes[ip][i].beta;
							assert (!Double.isNaN(nodes[ip][i].beta));
							nodes[ip][i].beta = sumNegLogProb (nodes[ip][i].beta,
																								 destinationNode.beta + transitionCost);
							assert (!Double.isNaN(nodes[ip][i].beta))
								: "dest.beta="+destinationNode.beta+" trans="+transitionCost+" sum="+(destinationNode.beta+transitionCost)
								+ " oldBeta="+oldBeta;
							xis[ip][i][j] = nodes[ip][i].alpha + transitionCost + nodes[ip+1][j].beta - cost;
							assert (!Double.isNaN(nodes[ip][i].alpha));
							assert (!Double.isNaN(transitionCost));
							assert (!Double.isNaN(nodes[ip+1][j].beta));
							assert (!Double.isNaN(cost));
							if (increment || outputAlphabet != null) {
								double p = Math.exp(-xis[ip][i][j]);
								assert (p < INFINITE_COST && !Double.isNaN(p)) : "xis["+ip+"]["+i+"]["+j+"]="+-xis[ip][i][j];
								if (increment)
									iter.incrementCount (p);
								if (outputAlphabet != null) {
									int outputIndex = outputAlphabet.lookupIndex (iter.getOutput(), false);
									assert (outputIndex >= 0);
									// xxx This assumes that "ip" == "op"!
									outputCounts[ip][outputIndex] += p;
									//System.out.println ("CRF Lattice outputCounts["+ip+"]["+outputIndex+"]+="+p);
								}
							}
						}
					}
					gammas[ip][i] = nodes[ip][i].alpha + nodes[ip][i].beta - cost;
				}
			}
			if (increment) 
				for (int i = 0; i < numStates; i++) {
					double p = Math.exp(-gammas[0][i]);
					assert (p < INFINITE_COST && !Double.isNaN(p));
					getState(i).incrementInitialCount (p);
				}
			if (outputAlphabet != null) {
				labelings = new LabelVector[latticeLength];
				for (int ip = latticeLength-2; ip >= 0; ip--) {
					assert (Math.abs(1.0-DenseVector.sum (outputCounts[ip])) < 0.000001);;
					labelings[ip] = new LabelVector (outputAlphabet, outputCounts[ip]);
				}
			}
		}

		public double getCost () {
			assert (!Double.isNaN(cost));
			return cost; }

		// No, this.cost is an "unnormalized cost"
		//public double getProbability () { return Math.exp (-cost); }

		public double getGammaCost (int inputPosition, State s) {
			return gammas[inputPosition][s.getIndex()]; }

		public double getGammaProbability (int inputPosition, State s) {
			return Math.exp (-gammas[inputPosition][s.getIndex()]); }

		public LabelVector getLabelingAtPosition (int outputPosition)	{
			if (labelings != null)
				return labelings[outputPosition];
			return null;
		}

		// xxx We are a non-static inner class so this should be easy; but how?
		public Transducer getTransducer () {
			throw new UnsupportedOperationException (); }


		// A container for some information about a particular input position and state
		private class LatticeNode
		{
			int inputPosition;
			// outputPosition not really needed until we deal with asymmetric epsilon.
			State state;
			Object output;
			double alpha = INFINITE_COST;
			double beta = INFINITE_COST;
			LatticeNode (int inputPosition, State state)	{
				this.inputPosition = inputPosition;
				this.state = state;
				assert (this.alpha == INFINITE_COST);	// xxx Remove this check
			}
		}
		
	}	// end of class Lattice


	
	public ViterbiPath viterbiPath (Object unpipedObject)
	{
		Instance carrier = new Instance (unpipedObject, null, null, null, inputPipe);
		return viterbiPath ((Sequence)carrier.getData());
	}
	
	public ViterbiPath viterbiPath (Sequence inputSequence)
	{
		return viterbiPath (inputSequence, null);
	}

	public ViterbiPath viterbiPath (Sequence inputSequence, Sequence outputSequence)
	{
		// xxx We don't do epsilon transitions for now
		assert (outputSequence == null
						|| inputSequence.size() == outputSequence.size());
		return new ViterbiPath (inputSequence, outputSequence);
	}


	public class ViterbiPath extends SequencePairAlignment
	{
		// double cost inherited from SequencePairAlignment
		// Sequence input, output inherited from SequencePairAlignment
		// this.output stores the actual output of Viterbi transitions
		Sequence providedOutput;
		ViterbiNode[] nodePath;
		int latticeLength;

		protected ViterbiNode getViterbiNode (ViterbiNode[][] nodes, int ip, int stateIndex)
		{
			if (nodes[ip][stateIndex] == null)
				nodes[ip][stateIndex] = new ViterbiNode (ip, getState (stateIndex));
			return nodes[ip][stateIndex];
		}

		// You may pass null for output
		protected ViterbiPath (Sequence inputSequence, Sequence outputSequence)
		{
			assert (inputSequence != null);
			if (logger.isLoggable (Level.FINE)) {
				logger.fine ("Starting ViterbiPath");
				logger.fine ("Input: ");
				for (int ip = 0; ip < inputSequence.size(); ip++)
					logger.fine (" " + inputSequence.get(ip));
				logger.fine ("\nOutput: ");
				if (outputSequence == null)
					logger.fine ("null");
				else
					for (int op = 0; op < outputSequence.size(); op++)
						logger.fine (" " + outputSequence.get(op));
				logger.fine ("\n");
			}

			this.input = inputSequence;
			this.providedOutput = outputSequence;
			// this.output is set at the end when we know the exact outputs
			// of the Viterbi path.  Note that in some cases the "output"
			// may be provided non-null as an argument to this method, but the
			// "output" objects may not be fully-specified even though they do
			// provide some restrictions.  This is why we set our this.output
			// from the outputs provided by the transition iterator along the
			// Viterbi path.

			// xxx Not very efficient when the lattice is actually sparse,
			// especially when the number of states is large and the
			// sequence is long.
			latticeLength = input.size()+1;
			int numStates = numStates();
			ViterbiNode[][] nodes = new ViterbiNode[latticeLength][numStates];

			// Viterbi Forward
			logger.fine ("Starting Viterbi");
			for (int i = 0; i < numStates; i++) {
				double initialCost = getState(i).initialCost;
				if (initialCost < INFINITE_COST) {
					ViterbiNode n = getViterbiNode (nodes, 0, i);
					n.delta = initialCost;
				}
			}
			for (int ip = 0; ip < latticeLength-1; ip++)
				for (int i = 0; i < numStates; i++) {
					if (nodes[ip][i] == null || nodes[ip][i].delta == INFINITE_COST)
						// xxx if we end up doing this a lot,
						// we could save a list of the non-null ones
						continue;
					State s = getState(i);
					TransitionIterator iter = s.transitionIterator (input, ip, providedOutput, ip);
					if (logger.isLoggable (Level.FINE))
						logger.fine (" Starting Viterbi transition iteration from state "
												 + s.getName() + " on input " + input.get(ip));
					while (iter.hasNext()) {
						State destination = iter.nextState();
						if (logger.isLoggable (Level.FINE))
							logger.fine ("Viterbi[inputPos="+ip
													 +"][source="+s.getName()
													 +"][dest="+destination.getName()+"]");
						ViterbiNode destinationNode = getViterbiNode (nodes, ip+1,
																													destination.getIndex());
						destinationNode.output = iter.getOutput();
						cost = nodes[ip][i].delta + iter.getCost();
						if (ip == latticeLength-2)
							cost += destination.getFinalCost();
						if (cost < destinationNode.delta) {
							if (logger.isLoggable (Level.FINE))
								logger.fine ("Viterbi[inputPos="+ip
														 +"][source][dest="+destination.getName()
														 +"] cost reduced to "+cost+" by source="+
														 s.getName());
							destinationNode.delta = cost;
							destinationNode.minCostPredecessor = nodes[ip][i];
						}
					}
				}

			// Find the final state with minimum cost, and get the total
			// cost of the Viterbi path.
			ViterbiNode minCostNode;
			int ip = latticeLength-1;
			this.cost = INFINITE_COST;
			minCostNode = null;
			for (int i = 0; i < numStates; i++) {
				if (nodes[ip][i] == null)
					continue;
				if (nodes[ip][i].delta < this.cost) {
					minCostNode = nodes[ip][i];
					this.cost = minCostNode.delta;
				}
			}

			// Build the path and the output sequence.
			this.nodePath = new ViterbiNode[latticeLength];
			Object[] outputArray = new Object[input.size()];
			for (ip = latticeLength-1; ip >= 0; ip--) {
				this.nodePath[ip] = minCostNode;
				if (ip > 0)
					outputArray[ip-1] = minCostNode.output;
				minCostNode = minCostNode.minCostPredecessor;
			}
			this.output = new ArraySequence (outputArray, false);
		}

		// xxx Delete this method and move it into the Viterbi constructor, just like Lattice.
		// Increment states and transitions in the tranducer with the
		// expectations from this path.  This provides for the so-called
		// "Viterbi training" approximation.
		public void incrementTransducerCounts ()
		{
			nodePath[0].state.incrementInitialCount (1.0);
			nodePath[nodePath.length-1].state.incrementFinalCount (1.0);
			for (int ip = 0; ip < nodePath.length-1; ip++) {
				TransitionIterator iter =
					nodePath[ip].state.transitionIterator (input, ip,
																								 providedOutput, ip);
				// xxx This assumes that a transition is completely
				// identified, and made unique by its destination state and
				// output.  This may not be true!
				int numIncrements = 0;
				while (iter.hasNext()) {
					if (iter.nextState().equals (nodePath[ip+1].state)
							&& iter.getOutput().equals (nodePath[ip].output)) {
						iter.incrementCount (1.0);
						numIncrements++;
					}
				}
				if (numIncrements > 1)
					throw new IllegalStateException ("More than one satisfying transition found.");
				if (numIncrements == 0)
					throw new IllegalStateException ("No satisfying transition found.");
			}
		}
		
		public SequencePairAlignment trimStateInfo ()
		{
			return new SequencePairAlignment (input, output, cost);
		}

		public double tokenAccuracy (Sequence referenceOutput)
		{
			int accuracy = 0;
			assert (referenceOutput.size() == output.size());
			for (int i = 0; i < output.size(); i++) {
				//logger.fine("tokenAccuracy: ref: "+referenceOutput.get(i)+" viterbi: "+output.get(i));
				if (referenceOutput.get(i).toString().equals (output.get(i).toString())) {
					accuracy++;
				}
			}
			logger.info ("Number correct: " + accuracy + " out of " + output.size());
			return ((double)accuracy)/output.size();
		}

		public double tokenAccuracy (Sequence referenceOutput, PrintWriter out)
		{
			int accuracy = 0;
			String testString;
			assert (referenceOutput.size() == output.size());
			for (int i = 0; i < output.size(); i++) {
				//logger.fine("tokenAccuracy: ref: "+referenceOutput.get(i)+" viterbi: "+output.get(i));
				testString = output.get(i).toString();
				if (out != null) {
					out.println(testString);
				}
				if (referenceOutput.get(i).toString().equals (testString)) {
					accuracy++;
				}
			}
			logger.info ("Number correct: " + accuracy + " out of " + output.size());
			return ((double)accuracy)/output.size();
		}

		
		private class ViterbiNode
		{
			int inputPosition;								// Position of input used to enter this node
			State state;											// Transducer state from which this node entered
			Object output;										// Transducer output produced on entering this node
			double delta = INFINITE_COST;
			ViterbiNode minCostPredecessor = null;
			ViterbiNode (int inputPosition, State state)
			{
				this.inputPosition = inputPosition;
				this.state = state;
			}
		}

	} // end of ViterbiPath


	
	/* sumNegLogProb()
		 
		 We need to be able to sum probabilities that are represented as
		 costs (which are -log(probabilities)).  Naively, we would just
		 convert them into probabilities, sum them, and then convert them
		 back into costs.  This would be:

		 double sumNegLogProb (double a, double b) {
		   return -Math.log (Math.exp(-a) + Math.exp(-b));
		 }

		 This is how this function was originally implemented, but it
		 fails when a or b is too negative.  The machine would have the
		 resolution to represent the final cost, but not the resolution to
		 represent the intermediate exponentiated negative costs, and we
		 would get -infinity as our answer.

		 What we want is a method for getting the sum by exponentiating a
		 number that is not too large.  We can do this with the following.
		 Starting with the equation above, then:

		 sumNegProb = -log (exp(-a) + exp(-b))
		 -sumNegProb = log (exp(-a) + exp(-b))
		 exp(-sumNegProb) = exp(-a) + exp(-b)
		 exp(-sumNegProb)/exp(-a) = 1 + exp(-b)/exp(-a)
		 exp(-sumNegProb+a) = 1 + exp(-b+a)
		 -sumNegProb+a = log (1 + exp(-b+a))
		 sumNegProb = a - log (1 + exp(a-b)).

		 We want to make sure that "a-b" is negative or a small positive
		 number.  We can assure this by noticing that we could have
		 equivalently derived

		 sumNegProb = b - log (1 + exp(b-a)),

		 and we can simply select among the two alternative equations the
		 one that would have the smallest (or most negative) exponent.

	*/

	private static double sumNegLogProb (double a, double b)
	{
		if (a == Double.POSITIVE_INFINITY && b == Double.POSITIVE_INFINITY)
			return Double.POSITIVE_INFINITY;
		else if (a > b)
			return b - Math.log (1 + Math.exp(b-a));
		else
			return a - Math.log (1 + Math.exp(a-b));
	}

		
}
