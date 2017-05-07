package org.tensorflow.op.math;

import org.tensorflow.DataType;
import org.tensorflow.InputSource;
import org.tensorflow.op.Scope;

public class MathOps {
  
  public MathOps(Scope scope) {
    this.scope = scope;
  }
  
  public Cast.Builder cast(InputSource x, DataType y) {
    return new Cast.Builder(scope, x, y);
  }

  private final Scope scope;
}
