package org.tensorflow.op;

import org.tensorflow.Graph;

class GraphSubBuilder extends GraphBuilder {

  GraphSubBuilder(Graph graph, Scope scope) {
    super(graph, scope);
  }
  
  @Override
  public Graph build() {
    throw new UnsupportedOperationException("Graph subbuilders cannot finalize the graph");
  }
}
