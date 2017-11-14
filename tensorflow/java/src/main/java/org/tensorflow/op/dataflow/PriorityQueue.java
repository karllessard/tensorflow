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

public final class PriorityQueue extends PrimitiveOp implements Operand<String> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param componentTypes
     **/
    public Options componentTypes(List<DataType> componentTypes) {
      this.componentTypes = componentTypes;
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
    
    private List<DataType> componentTypes;
    private Integer capacity;
    private String container;
    private String sharedName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new PriorityQueue operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shapes
   * @return a new instance of PriorityQueue
   **/
  public static PriorityQueue create(Scope scope, List<Shape> shapes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("PriorityQueue", scope.makeOpName("PriorityQueue"));
    opBuilder.setAttr("shapes", shapes.toArray(new Shape[shapes.size()]));
    return new PriorityQueue(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new PriorityQueue operation to the graph.
   * 
   * @param scope Current graph scope
   * @param shapes
   * @param options an object holding optional attributes values
   * @return a new instance of PriorityQueue
   **/
  public static PriorityQueue create(Scope scope, List<Shape> shapes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("PriorityQueue", scope.makeOpName("PriorityQueue"));
    opBuilder.setAttr("shapes", shapes.toArray(new Shape[shapes.size()]));
    if (options.componentTypes != null) {
      opBuilder.setAttr("componentTypes", options.componentTypes.toArray(new DataType[options.componentTypes.size()]));
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
    return new PriorityQueue(opBuilder.build());
  }
  
  /**
   * @param componentTypes
   **/
  public static Options componentTypes(List<DataType> componentTypes) {
    return new Options().componentTypes(componentTypes);
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
  
  public Output<String> handle() {
    return handle;
  }
  
  @Override
  public Output<String> asOutput() {
    return handle;
  }
  
  private Output<String> handle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private PriorityQueue(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
