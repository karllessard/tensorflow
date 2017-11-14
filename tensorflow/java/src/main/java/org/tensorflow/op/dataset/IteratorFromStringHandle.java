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
package org.tensorflow.op.dataset;

import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class IteratorFromStringHandle extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param outputTypes
     **/
    public Options outputTypes(List<DataType> outputTypes) {
      this.outputTypes = outputTypes;
      return this;
    }
    
    /**
     * @param outputShapes
     **/
    public Options outputShapes(List<Shape> outputShapes) {
      this.outputShapes = outputShapes;
      return this;
    }
    
    private List<DataType> outputTypes;
    private List<Shape> outputShapes;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new IteratorFromStringHandle operation to the graph.
   * 
   * @param scope Current graph scope
   * @param stringHandle
   * @return a new instance of IteratorFromStringHandle
   **/
  public static IteratorFromStringHandle create(Scope scope, Operand<String> stringHandle) {
    OperationBuilder opBuilder = scope.graph().opBuilder("IteratorFromStringHandle", scope.makeOpName("IteratorFromStringHandle"));
    opBuilder.addInput(stringHandle.asOutput());
    return new IteratorFromStringHandle(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new IteratorFromStringHandle operation to the graph.
   * 
   * @param scope Current graph scope
   * @param stringHandle
   * @param options an object holding optional attributes values
   * @return a new instance of IteratorFromStringHandle
   **/
  public static IteratorFromStringHandle create(Scope scope, Operand<String> stringHandle, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("IteratorFromStringHandle", scope.makeOpName("IteratorFromStringHandle"));
    opBuilder.addInput(stringHandle.asOutput());
    if (options.outputTypes != null) {
      opBuilder.setAttr("outputTypes", options.outputTypes.toArray(new DataType[options.outputTypes.size()]));
    }
    if (options.outputShapes != null) {
      opBuilder.setAttr("outputShapes", options.outputShapes.toArray(new Shape[options.outputShapes.size()]));
    }
    return new IteratorFromStringHandle(opBuilder.build());
  }
  
  /**
   * @param outputTypes
   **/
  public static Options outputTypes(List<DataType> outputTypes) {
    return new Options().outputTypes(outputTypes);
  }
  
  /**
   * @param outputShapes
   **/
  public static Options outputShapes(List<Shape> outputShapes) {
    return new Options().outputShapes(outputShapes);
  }
  
  public Output<?> resourceHandle() {
    return resourceHandle;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<Object> asOutput() {
    return (Output<Object>) resourceHandle;
  }
  
  private Output<?> resourceHandle;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private IteratorFromStringHandle(Operation operation) {
    super(operation);
    int outputIdx = 0;
    resourceHandle = operation.output(outputIdx++);
  }
}
