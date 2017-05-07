package org.tensorflow.op.math;

import org.tensorflow.DataType;
import org.tensorflow.InputSource;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.Scope;
import org.tensorflow.op.SingleResultNode;

public class Cast extends SingleResultNode {

  public static class Builder extends SingleResultNode.Builder<Cast> {

    Builder(Scope scope, Output x, DataType y) {
      super(scope, OP_TYPE);
      this.x = x;
      this.y = y;
    }

    @Override
    protected Cast buildOp(OperationBuilder opBuilder) {
      return new Cast(opBuilder
            .addInput(x)
            .setAttr("DstT", y)
            .build());
    }

    private final Output x;
    private final DataType y;
  }

  private Cast(Operation op) {
    super(op);
  }

  private static final String OP_TYPE = "Cast";
}
