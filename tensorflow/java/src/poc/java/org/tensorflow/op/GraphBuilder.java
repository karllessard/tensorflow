package org.tensorflow.op;

import org.tensorflow.op.image.ImageOps;
import org.tensorflow.op.std.Constant;
import org.tensorflow.op.std.StdOps;

public class GraphBuilder implements AutoCloseable {
  
  private class Api {

    Api(Scope scope) {
      stdOps = new StdOps(scope);
      imageOps = new ImageOps(scope);
    }
    
    private StdOps stdOps;
    private ImageOps imageOps;
  }

  public GraphBuilder() {
    graph = new Graph2();
    api = new Api(new Scope(graph));
  }

  protected GraphBuilder(Graph2 graph, Scope scope) {
    this.graph = graph;
    api = new Api(scope);
  }
  
  public ImageOps image() {
    return api.imageOps;
  }
  
  // Exposes STD operations directly from the GraphBuilder
  public Constant.Builder constant(Object value) {
    return api.stdOps.constant(value);
  }
  
  public GraphBuilder withSubscope(String prefix) {
    return new GraphSubBuilder(graph, new Scope(graph, prefix));
  }
  
  public Graph2 build() {
    if (graph.root() == null) {
      throw new IllegalStateException("Graph has no root node");
    }
    return graph;
  }

  @Override
  public void close() throws Exception {
    graph.close();
  }
  
  private final Graph2 graph;
  private final Api api;
}
