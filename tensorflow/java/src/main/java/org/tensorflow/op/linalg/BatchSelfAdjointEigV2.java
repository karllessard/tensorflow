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
package org.tensorflow.op.linalg;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class BatchSelfAdjointEigV2<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param computeV
     **/
    public Options computeV(Boolean computeV) {
      this.computeV = computeV;
      return this;
    }
    
    private Boolean computeV;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new BatchSelfAdjointEigV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @return a new instance of BatchSelfAdjointEigV2
   **/
  public static <T> BatchSelfAdjointEigV2<T> create(Scope scope, Operand<T> input) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchSelfAdjointEigV2", scope.makeOpName("BatchSelfAdjointEigV2"));
    opBuilder.addInput(input.asOutput());
    return new BatchSelfAdjointEigV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new BatchSelfAdjointEigV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param options an object holding optional attributes values
   * @return a new instance of BatchSelfAdjointEigV2
   **/
  public static <T> BatchSelfAdjointEigV2<T> create(Scope scope, Operand<T> input, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchSelfAdjointEigV2", scope.makeOpName("BatchSelfAdjointEigV2"));
    opBuilder.addInput(input.asOutput());
    if (options.computeV != null) {
      opBuilder.setAttr("computeV", options.computeV);
    }
    return new BatchSelfAdjointEigV2<T>(opBuilder.build());
  }
  
  /**
   * @param computeV
   **/
  public static Options computeV(Boolean computeV) {
    return new Options().computeV(computeV);
  }
  
  public Output<T> e() {
    return e;
  }
  
  public Output<T> v() {
    return v;
  }
  
  private Output<T> e;
  private Output<T> v;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private BatchSelfAdjointEigV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    e = operation.output(outputIdx++);
    v = operation.output(outputIdx++);
  }
}
