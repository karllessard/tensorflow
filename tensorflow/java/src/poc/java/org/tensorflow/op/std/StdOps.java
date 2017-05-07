package org.tensorflow.op.std;

import org.tensorflow.op.Scope;

public class StdOps {
  
  public StdOps(Scope scope) {
    this.scope = scope;
  }

  public Constant.Builder constant(Object value) {
    return new Constant.Builder(scope, value);
  }

  private final Scope scope;
}
