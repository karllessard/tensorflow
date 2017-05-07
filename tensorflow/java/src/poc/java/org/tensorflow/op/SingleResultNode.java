package org.tensorflow.op;

import org.tensorflow.InputSource;
import org.tensorflow.Operation;
import org.tensorflow.Output;

public abstract class SingleResultNode extends Node implements InputSource {
  
  protected abstract static class Builder<T extends SingleResultNode> extends NodeBuilder<T> implements InputSource {

    protected Builder(Scope scope, String opType) {
      super(scope, opType);
    }

    @Override
    public Output input() {
      return build().input();
    }
  }

  protected SingleResultNode(Operation op) {
    super(op);
  }

  @Override
  public Output input() {
    return op().output(0);
  }
}
