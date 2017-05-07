package org.tensorflow.op;

class GraphSubBuilder extends GraphBuilder {

  GraphSubBuilder(Graph2 graph, Scope scope) {
    super(graph, scope);
  }
  
  @Override
  public Graph2 build() {
    throw new UnsupportedOperationException("Graph subbuilders cannot finalize the graph");
  }
}
