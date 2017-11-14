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

public final class SparseApplyRMSProp<T> extends PrimitiveOp implements Operand<T> {
  
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
   * Factory method to create a class to wrap a new SparseApplyRMSProp operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param ms
   * @param mom
   * @param lr
   * @param rho
   * @param momentum
   * @param epsilon
   * @param grad
   * @param indices
   * @return a new instance of SparseApplyRMSProp
   **/
  public static <T, U> SparseApplyRMSProp<T> create(Scope scope, Operand<T> var, Operand<T> ms, Operand<T> mom, Operand<T> lr, Operand<T> rho, Operand<T> momentum, Operand<T> epsilon, Operand<T> grad, Operand<U> indices) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseApplyRMSProp", scope.makeOpName("SparseApplyRMSProp"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(ms.asOutput());
    opBuilder.addInput(mom.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(rho.asOutput());
    opBuilder.addInput(momentum.asOutput());
    opBuilder.addInput(epsilon.asOutput());
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    return new SparseApplyRMSProp<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new SparseApplyRMSProp operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param ms
   * @param mom
   * @param lr
   * @param rho
   * @param momentum
   * @param epsilon
   * @param grad
   * @param indices
   * @param options an object holding optional attributes values
   * @return a new instance of SparseApplyRMSProp
   **/
  public static <T, U> SparseApplyRMSProp<T> create(Scope scope, Operand<T> var, Operand<T> ms, Operand<T> mom, Operand<T> lr, Operand<T> rho, Operand<T> momentum, Operand<T> epsilon, Operand<T> grad, Operand<U> indices, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("SparseApplyRMSProp", scope.makeOpName("SparseApplyRMSProp"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(ms.asOutput());
    opBuilder.addInput(mom.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(rho.asOutput());
    opBuilder.addInput(momentum.asOutput());
    opBuilder.addInput(epsilon.asOutput());
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    if (options.useLocking != null) {
      opBuilder.setAttr("useLocking", options.useLocking);
    }
    return new SparseApplyRMSProp<T>(opBuilder.build());
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
  private SparseApplyRMSProp(Operation operation) {
    super(operation);
    int outputIdx = 0;
    out = operation.output(outputIdx++);
  }
}
