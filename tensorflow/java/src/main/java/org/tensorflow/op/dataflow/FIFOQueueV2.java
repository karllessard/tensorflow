/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.op.dataflow;

import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class FIFOQueueV2 extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param shapes
     **/
    public Options shapes(List<Shape> shapes) {
      this.shapes = shapes;
      return this;
    }
    
    /**
     * @param capacity
     **/
    public Options capacity(Integer capacity) {
      this.capacity = capacity;
      return this;
    }
    
    /**
     * @param container
     **/
    public Options container(String container) {
      this.container = container;
      return this;
    }
    
    /**
     * @param sharedName
     **/
    public Options sharedName(String sharedName) {
      this.sharedName = sharedName;
      return this;
    }
    
    private List<Shape> shapes;
    private Integer capacity;
    private String container;
    private String sharedName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FIFOQueueV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param componentTypes
   * @return a new instance of FIFOQueueV2
   **/
  public static FIFOQueueV2 create(Scope scope, List<DataType> componentTypes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FIFOQueueV2", scope.makeOpName("FIFOQueueV2"));
    opBuilder.setAttr("componentTypes", componentTypes.toArray(new DataType[componentTypes.size()]));
    return new FIFOQueueV2(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FIFOQueueV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param componentTypes
   * @param options an object holding optional attributes values
   * @return a new instance of FIFOQueueV2
   **/
  public static FIFOQueueV2 create(Scope scope, List<DataType> componentTypes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FIFOQueueV2", scope.makeOpName("FIFOQueueV2"));
    opBuilder.setAttr("componentTypes", componentTypes.toArray(new DataType[componentTypes.size()]));
    if (options.shapes != null) {
      opBuilder.setAttr("shapes", options.shapes.toArray(new Shape[options.shapes.size()]));
    }
    if (options.capacity != null) {
      opBuilder.setAttr("capacity", options.capacity);
    }
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    return new FIFOQueueV2(opBuilder.build());
  }
  
  /**
   * @param shapes
   **/
  public static Options shapes(List<Shape> shapes) {
    return new Options().shapes(shapes);
  }
  
  /**
   * @param capacity
   **/
  public static Options capacity(Integer capacity) {
    return new Options().capacity(capacity);
  }
  
  /**
   * @param container
   **/
  public static Options container(String container) {
    return new Options().container(container);
  }
  
  /**
   * @param sharedName
   **/
  public static Options sharedName(String sharedName) {
    return new Options().sharedName(sharedName);
  }
  
  public Output<?> handle() {
    return handle;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<Object> asOutput() {
    return (Output<Object>) handle;
  }
  
  private Output<?> handle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FIFOQueueV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
