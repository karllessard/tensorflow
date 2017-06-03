package org.tensorflow;

/**
 * A specialization of the {@link Output} for variable tensors.
 *
 * <p>The main purpose of this class if to be able to pass an output to an operation that takes a
 * {@link VarInput} as an operand.
 *
 * @see {@link Output}
 */
public class VarOutput extends Output implements VarInput {

  /** Handle to the idx-th variable output of the Operation {@code op}. */
  public VarOutput(Operation op, int idx) {
    super(op, idx);
  }
}
