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

public final class OneShotIterator extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
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
    
    private String container;
    private String sharedName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new OneShotIterator operation to the graph.
   * 
   * @param scope Current graph scope
   * @param datasetFactory
   * @param outputTypes
   * @param outputShapes
   * @return a new instance of OneShotIterator
   **/
  public static OneShotIterator create(Scope scope, Object datasetFactory, List<DataType> outputTypes, List<Shape> outputShapes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("OneShotIterator", scope.makeOpName("OneShotIterator"));
    opBuilder.setAttr("datasetFactory", datasetFactory);
    opBuilder.setAttr("outputTypes", outputTypes.toArray(new DataType[outputTypes.size()]));
    opBuilder.setAttr("outputShapes", outputShapes.toArray(new Shape[outputShapes.size()]));
    return new OneShotIterator(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new OneShotIterator operation to the graph.
   * 
   * @param scope Current graph scope
   * @param datasetFactory
   * @param outputTypes
   * @param outputShapes
   * @param options an object holding optional attributes values
   * @return a new instance of OneShotIterator
   **/
  public static OneShotIterator create(Scope scope, Object datasetFactory, List<DataType> outputTypes, List<Shape> outputShapes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("OneShotIterator", scope.makeOpName("OneShotIterator"));
    opBuilder.setAttr("datasetFactory", datasetFactory);
    opBuilder.setAttr("outputTypes", outputTypes.toArray(new DataType[outputTypes.size()]));
    opBuilder.setAttr("outputShapes", outputShapes.toArray(new Shape[outputShapes.size()]));
    if (options.container != null) {
      opBuilder.setAttr("container", options.container);
    }
    if (options.sharedName != null) {
      opBuilder.setAttr("sharedName", options.sharedName);
    }
    return new OneShotIterator(opBuilder.build());
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
  private OneShotIterator(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
