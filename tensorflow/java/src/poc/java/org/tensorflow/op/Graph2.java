package org.tensorflow.op;

import org.tensorflow.Graph;

public class Graph2 extends Graph {
  
  public Node root() {
    return root;
  }

  public String rootName() {
    return root.op().name();
  }

  void root(Node root) {
    if (root != null) {
      throw new IllegalStateException("Graph can only have a single root node");
    }
    this.root = root;
  }
  
  private Node root;
}
