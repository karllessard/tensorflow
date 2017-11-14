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
package org.tensorflow.op.string;

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class StringToHashBucketStrong extends PrimitiveOp implements Operand<Long> {
  
  /**
   * Factory method to create a class to wrap a new StringToHashBucketStrong operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param numBuckets
   * @param key
   * @return a new instance of StringToHashBucketStrong
   **/
  public static StringToHashBucketStrong create(Scope scope, Operand<String> input, Integer numBuckets, List<Integer> key) {
    OperationBuilder opBuilder = scope.graph().opBuilder("StringToHashBucketStrong", scope.makeOpName("StringToHashBucketStrong"));
    opBuilder.addInput(input.asOutput());
    opBuilder.setAttr("numBuckets", numBuckets);
    long[] keyArray = new long[key.size()];
    for (int i = 0; i < keyArray.length; ++i) {
      keyArray[i] = key.get(i);
    }
    opBuilder.setAttr("key", keyArray);
    return new StringToHashBucketStrong(opBuilder.build());
  }
  
  public Output<Long> output() {
    return output;
  }
  
  @Override
  public Output<Long> asOutput() {
    return output;
  }
  
  private Output<Long> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private StringToHashBucketStrong(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
