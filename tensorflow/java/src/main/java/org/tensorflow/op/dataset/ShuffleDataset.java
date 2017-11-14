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

public final class ShuffleDataset extends PrimitiveOp implements Operand<Object> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param reshuffleEachIteration
     **/
    public Options reshuffleEachIteration(Boolean reshuffleEachIteration) {
      this.reshuffleEachIteration = reshuffleEachIteration;
      return this;
    }
    
    private Boolean reshuffleEachIteration;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ShuffleDataset operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputDataset
   * @param bufferSize
   * @param seed
   * @param seed2
   * @param outputTypes
   * @param outputShapes
   * @return a new instance of ShuffleDataset
   **/
  public static ShuffleDataset create(Scope scope, Operand<?> inputDataset, Operand<Long> bufferSize, Operand<Long> seed, Operand<Long> seed2, List<DataType> outputTypes, List<Shape> outputShapes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ShuffleDataset", scope.makeOpName("ShuffleDataset"));
    opBuilder.addInput(inputDataset.asOutput());
    opBuilder.addInput(bufferSize.asOutput());
    opBuilder.addInput(seed.asOutput());
    opBuilder.addInput(seed2.asOutput());
    opBuilder.setAttr("outputTypes", outputTypes.toArray(new DataType[outputTypes.size()]));
    opBuilder.setAttr("outputShapes", outputShapes.toArray(new Shape[outputShapes.size()]));
    return new ShuffleDataset(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ShuffleDataset operation to the graph.
   * 
   * @param scope Current graph scope
   * @param inputDataset
   * @param bufferSize
   * @param seed
   * @param seed2
   * @param outputTypes
   * @param outputShapes
   * @param options an object holding optional attributes values
   * @return a new instance of ShuffleDataset
   **/
  public static ShuffleDataset create(Scope scope, Operand<?> inputDataset, Operand<Long> bufferSize, Operand<Long> seed, Operand<Long> seed2, List<DataType> outputTypes, List<Shape> outputShapes, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ShuffleDataset", scope.makeOpName("ShuffleDataset"));
    opBuilder.addInput(inputDataset.asOutput());
    opBuilder.addInput(bufferSize.asOutput());
    opBuilder.addInput(seed.asOutput());
    opBuilder.addInput(seed2.asOutput());
    opBuilder.setAttr("outputTypes", outputTypes.toArray(new DataType[outputTypes.size()]));
    opBuilder.setAttr("outputShapes", outputShapes.toArray(new Shape[outputShapes.size()]));
    if (options.reshuffleEachIteration != null) {
      opBuilder.setAttr("reshuffleEachIteration", options.reshuffleEachIteration);
    }
    return new ShuffleDataset(opBuilder.build());
  }
  
  /**
   * @param reshuffleEachIteration
   **/
  public static Options reshuffleEachIteration(Boolean reshuffleEachIteration) {
    return new Options().reshuffleEachIteration(reshuffleEachIteration);
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
  private ShuffleDataset(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
