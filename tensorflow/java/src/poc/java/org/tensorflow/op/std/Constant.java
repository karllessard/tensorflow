package org.tensorflow.op.std;

import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.SingleResultNode;

public class Constant extends SingleResultNode {

  public static class Builder extends SingleResultNode.Builder<Constant> {

    Builder(Scope scope, Object value) {
      super(scope, OP_TYPE);
      this.value = value;
    }

    @Override
    protected Constant buildOp(OperationBuilder opBuilder) {
      try (Tensor t = Tensor.create(value)) {
        return new Constant(opBuilder
            .setAttr("value", t)
            .setAttr("dtype", t.dataType())
            .build());
      }
    }
    
    private final Object value;
  }

  private Constant(Operation op) {
    super(op);
  }

  private static final String OP_TYPE = "Const";
}
