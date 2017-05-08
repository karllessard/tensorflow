package org.tensorflow.op.test;

import org.junit.Test;
import org.tensorflow.Session;
import org.tensorflow.op.Graph2;
import org.tensorflow.op.GraphBuilder;
import org.tensorflow.op.std.Constant;

public class GraphBuilderTest {
  
  private byte[] testImageBytes = {}; // TODO
  
  @Test
  public void testBuildingSingleResultNode() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.constant(testImageBytes);
    }
  }

  @Test
  public void testBuildingSingleResultNodeWithName() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.constant(testImageBytes).op("image");
    }
  }

  @Test
  public void testPassingSingleResultNodeBuilderAsAnInput() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.image().decodeJpeg(gb.constant(testImageBytes));
    }
  }

  @Test
  public void testPassingSingleResultNodeAsAnInput() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      Constant image = gb.constant(testImageBytes).op("image");
      gb.image().decodeJpeg(image);
    }
  }

  @Test
  public void testBuildingNodeWithOptionalAttributes() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.image().decodeJpeg(gb.constant(testImageBytes)).withChannels(3);
    }
  }

  @Test(expected = IllegalStateException.class)
  public void testBuildGraphWithoutRoot() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.image().decodeJpeg(gb.constant(testImageBytes));
      gb.build();
    }
  }

  @Test
  public void testBuildGraph() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.withRoot(
           gb.image().decodeJpeg(gb.constant(testImageBytes)).op()
      );
      gb.build();
    }
  }
  
  @Test
  public void testRunSessionFromGraphRoot() throws Exception {
    try (GraphBuilder gb = new GraphBuilder()) {
      gb.withRoot(
          gb.image().decodeJpeg(gb.constant(testImageBytes)).op()
      );
      Graph2 graph = gb.build();

      try (Session ss = new Session(graph)) {
        ss.runner().fetch(graph.rootName()).run();
      }
    }
  }
}
