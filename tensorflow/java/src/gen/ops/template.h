/*
 * template.h
 *
 *  Created on: Feb 12, 2017
 *      Author: karl
 */

#ifndef SRC_GEN_OPS_TEMPLATE_H_
#define SRC_GEN_OPS_TEMPLATE_H_

#include <string>
#include <list>

#include "tensorflow/core/platform/env.h"
#include "writer.h"

namespace tensorflow {

class Template {

public:
  struct Params : std::map<const string, string> {
    std::map<const string, std::list<Params>> groups;
  };

  Template();
  virtual ~Template();

  void Load(Env* env, const std::string& name);
  string Render(const Params& params);
  void RenderToFile(const string& fname, const Params& params);

private:
  Env* env;
  string name;
  string template_src;

  Template(Env* env, const string& template_src);

  void RenderWith(Writer& writer, const Params& params);
  size_t ProcessParam(Writer& writer, size_t param_pos, const Params& params);
  size_t ProcessInnerTemplateParam(Writer& writer, const std::string& param_name, size_t src_pos, const Params& params);
};

}

#endif /* SRC_GEN_OPS_TEMPLATE_H_ */
