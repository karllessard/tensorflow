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
package org.tensorflow.op.array;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Unpack<T> extends PrimitiveOp implements Iterable<Operand<T>> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param axis
     **/
    public Options axis(Integer axis) {
      this.axis = axis;
      return this;
    }
    
    private Integer axis;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Unpack operation to the graph.
   * 
   * @param scope Current graph scope
   * @param value
   * @param num
   * @return a new instance of Unpack
   **/
  public static <T> Unpack<T> create(Scope scope, Operand<T> value, Integer num) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Unpack", scope.makeOpName("Unpack"));
    opBuilder.addInput(value.asOutput());
    opBuilder.setAttr("num", num);
    return new Unpack<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Unpack operation to the graph.
   * 
   * @param scope Current graph scope
   * @param value
   * @param num
   * @param options an object holding optional attributes values
   * @return a new instance of Unpack
   **/
  public static <T> Unpack<T> create(Scope scope, Operand<T> value, Integer num, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Unpack", scope.makeOpName("Unpack"));
    opBuilder.addInput(value.asOutput());
    opBuilder.setAttr("num", num);
    if (options.axis != null) {
      opBuilder.setAttr("axis", options.axis);
    }
    return new Unpack<T>(opBuilder.build());
  }
  
  /**
   * @param axis
   **/
  public static Options axis(Integer axis) {
    return new Options().axis(axis);
  }
  
  public List<Output<T>> output() {
    return output;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<T>> iterator() {
    return (Iterator) output.iterator();
  }
  
  private List<Output<T>> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private Unpack(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int outputLength = operation.outputListLength("output");
    output = Arrays.asList((Output<T>[])operation.outputList(outputIdx, outputLength));
    outputIdx += outputLength;
  }
}
