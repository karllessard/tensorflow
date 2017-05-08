package org.tensorflow.op;

import org.tensorflow.OperationBuilder;

public abstract class NodeBuilder<T extends Node> {

    protected NodeBuilder(Scope scope, String opType) {
      this.scope = scope;
      this.opType = opType;
    }

    public T op() {
      return op(opType);
    }

    public T op(String name) {
      OperationBuilder opBuilder = scope.opBuilder(opType, name);
      try {
        return buildOp(opBuilder);

      } catch (Throwable t) {
        opBuilder.build(); // TODO make OperationBuilder AutoCloseable instead
        throw t;
      }
    }

    protected abstract T buildOp(OperationBuilder opBuilder);
    
    private final Scope scope;
    private final String opType;
}
