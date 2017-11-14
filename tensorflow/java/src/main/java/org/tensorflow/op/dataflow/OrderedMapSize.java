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
import org.tensorflow.op.Scope;

public final class OrderedMapSize extends PrimitiveOp implements Operand<Integer> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param capacity
     **/
    public Options capacity(Integer capacity) {
      this.capacity = capacity;
      return this;
    }
    
    /**
     * @param memoryLimit
     **/
    public Options memoryLimit(Integer memoryLimit) {
      this.memoryLimit = memoryLimit;
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
    
    private Integer capacity;
    private Integer memoryLimit;
    private String container;
    private String sharedName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new OrderedMapSize operation to the graph.
   * 
   * @param scope Current graph scope
   * @param dtypes
   * @return a new instance of OrderedMapSize
   **/
  public static OrderedMapSize create(Scope scope, List<DataType> dtypes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("OrderedMapSize", scope.makeOpName("OrderedMapSize"));
    opBuilder.setAttr("dtypes", dtypes.toArray(new DataType[dtypes.size()]));
    return new OrderedMapSize(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new OrderedMapSize operation to the graph.
   * 
   * @param scope Current graph scope
   * @param dtypes
   * @param options an object holding optional attributes values
   * @return a new instance of OrderedMapSize
   **/
  public static OrderedMapSize create(Scope scope, List<DataType> dtypes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("OrderedMapSize", scope.makeOpName("OrderedMapSize"));
    opBuilder.setAttr("dtypes", dtypes.toArray(new DataType[dtypes.size()]));
    if (options.capacity != null) {
      opBuilder.setAttr("capacity", options.capacity);
    }
    if (options.memoryLimit != null) {
      opBuilder.setAttr("memoryLimit", options.memoryLimit);
    }
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    return new OrderedMapSize(opBuilder.build());
  }
  
  /**
   * @param capacity
   **/
  public static Options capacity(Integer capacity) {
    return new Options().capacity(capacity);
  }
  
  /**
   * @param memoryLimit
   **/
  public static Options memoryLimit(Integer memoryLimit) {
    return new Options().memoryLimit(memoryLimit);
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
  
  public Output<Integer> size() {
    return size;
  }
  
  @Override
  public Output<Integer> asOutput() {
    return size;
  }
  
  private Output<Integer> size;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private OrderedMapSize(Operation operation) {
    super(operation);
    int outputIdx = 0;
    size = operation.output(outputIdx++);
  }
}
