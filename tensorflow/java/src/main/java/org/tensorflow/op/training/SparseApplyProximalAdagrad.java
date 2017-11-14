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
package org.tensorflow.op.training;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class SparseApplyProximalAdagrad<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param useLocking
     **/
    public Options useLocking(Boolean useLocking) {
      this.useLocking = useLocking;
      return this;
    }
    
    private Boolean useLocking;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new SparseApplyProximalAdagrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param accum
   * @param lr
   * @param l1
   * @param l2
   * @param grad
   * @param indices
   * @return a new instance of SparseApplyProximalAdagrad
   **/
  public static <T, U> SparseApplyProximalAdagrad<T> create(Scope scope, Operand<T> var, Operand<T> accum, Operand<T> lr, Operand<T> l1, Operand<T> l2, Operand<T> grad, Operand<U> indices) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseApplyProximalAdagrad", scope.makeOpName("SparseApplyProximalAdagrad"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(accum.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(l1.asOutput());
    opBuilder.addInput(l2.asOutput());
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    return new SparseApplyProximalAdagrad<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SparseApplyProximalAdagrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param accum
   * @param lr
   * @param l1
   * @param l2
   * @param grad
   * @param indices
   * @param options an object holding optional attributes values
   * @return a new instance of SparseApplyProximalAdagrad
   **/
  public static <T, U> SparseApplyProximalAdagrad<T> create(Scope scope, Operand<T> var, Operand<T> accum, Operand<T> lr, Operand<T> l1, Operand<T> l2, Operand<T> grad, Operand<U> indices, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseApplyProximalAdagrad", scope.makeOpName("SparseApplyProximalAdagrad"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(accum.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(l1.asOutput());
    opBuilder.addInput(l2.asOutput());
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    if (options.useLocking != null) {
      opBuilder.setAttr("useLocking", options.useLocking);
    }
    return new SparseApplyProximalAdagrad<T>(opBuilder.build());
  }
  
  /**
   * @param useLocking
   **/
  public static Options useLocking(Boolean useLocking) {
    return new Options().useLocking(useLocking);
  }
  
  public Output<T> out() {
    return out;
  }
  
  @Override
  public Output<T> asOutput() {
    return out;
  }
  
  private Output<T> out;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private SparseApplyProximalAdagrad(Operation operation) {
    super(operation);
    int outputIdx = 0;
    out = operation.output(outputIdx++);
  }
}
