package org.tensorflow.op;

import org.tensorflow.OperationBuilder;

public class Scope {
  
  public Scope(Graph2 graph) {
    this(graph, "");
  }

  public Scope(Graph2 graph, String prefix) {
    super();
    this.graph = graph;
    this.prefix = prefix;
  }

  OperationBuilder opBuilder(String type, String name) {
    return graph.opBuilder(type, prefix + name);
  }
  
  Graph2 graph() {
    return graph;
  }

  private final Graph2 graph;
  private final String prefix;
}
