package org.tensorflow;

/**
 * A specialization of the {@link Output} for variable tensors.
 *
 * <p>The main purpose of this class if to be able to pass an output to an operation that takes a
 * {@link VariableInput} as an operand.
 *
 * @see {@link Output}
 */
public class VariableOutput extends Output implements VariableInput {

  /** Handle to the idx-th variable output of the Operation {@code op}. */
  public VariableOutput(Operation op, int idx) {
    super(op, idx);
  }
}
