package org.tensorflow.op;

import org.tensorflow.Operation;

public abstract class Node {
  
  protected Node(Operation op) {
    this.op = op;
  }

  public Operation op() {
    return op;
  }

  private final Operation op;
}
