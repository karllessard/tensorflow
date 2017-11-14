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

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class DynamicPartition<T> extends PrimitiveOp implements Iterable<Operand<T>> {
  
  /**
   * Factory method to create a class to wrap a new DynamicPartition operation to the graph.
   * 
   * @param scope Current graph scope
   * @param data
   * @param partitions
   * @param numPartitions
   * @return a new instance of DynamicPartition
   **/
  public static <T> DynamicPartition<T> create(Scope scope, Operand<T> data, Operand<Integer> partitions, Integer numPartitions) {
    OperationBuilder opBuilder = scope.graph().opBuilder("DynamicPartition", scope.makeOpName("DynamicPartition"));
    opBuilder.addInput(data.asOutput());
    opBuilder.addInput(partitions.asOutput());
    opBuilder.setAttr("numPartitions", numPartitions);
    return new DynamicPartition<T>(opBuilder.build());
  }
  
  public List<Output<T>> outputs() {
    return outputs;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<T>> iterator() {
    return (Iterator) outputs.iterator();
  }
  
  private List<Output<T>> outputs;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private DynamicPartition(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int outputsLength = operation.outputListLength("outputs");
    outputs = Arrays.asList((Output<T>[])operation.outputList(outputIdx, outputsLength));
    outputIdx += outputsLength;
  }
}
