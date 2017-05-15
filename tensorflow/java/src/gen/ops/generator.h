/*
 * op_generator.h
 *
 *  Created on: Feb 12, 2017
 *      Author: karl
 */

#ifndef SRC_GEN_OPS_GENERATOR_H_
#define SRC_GEN_OPS_GENERATOR_H_

#include "template.h"
#include "types.h"

namespace tensorflow {

class OpGenerator {

public:
  OpGenerator(const std::string& lib_fname);
  virtual ~OpGenerator();

  void Run(Env* env, bool include_internal);

private:
  const std::string lib_name;
  const std::string pkg_name;
  const std::string file_path;
  Template template_op;

  void WriteOps(const OpList& ops);
  void WriteOp(const OpDef& op, Template::Params& params);
  bool ImportType(const Type& type, Template::Params& params);
  std::string EscapeDoc(const std::string& src, int indent);

  inline string ToFileName(string base_name) {
    return file_path + base_name + ".java";
  }
};

} // namespace tensorflow

#endif /* SRC_GEN_OPS_GENERATOR_H_ */
