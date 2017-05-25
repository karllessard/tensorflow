package org.tensorflow;

public class OutputList implements InputList {

  public OutputList(Operation op, int start, String name) {
    int len = op.outputListLength(name);
    array = new Output[len];

    int end = start + len;
    for (int i = start; i < end; i++) {
      array[i - start] = op.output(i);
    }
  }

  @Override
  public Output[] toArray() {
    return array;
  }
  
  private final Output[] array;
}
