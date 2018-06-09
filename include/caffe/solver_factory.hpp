/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

/*简要说明：slover是什么？solver是caffe中实现训练模型参数更新的优化算法，solver类派生出的类可以对整个网络进行训练。在caffe中有很多solver子类，即不同的优化算法，如随机梯度下降（SGD）。
  一个solver factory可以注册solvers，运行时，注册过的solvers通过SolverRegistry::CreateSolver(param)来调用*/




#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
     //Creator为函数指针类型；CreatorRegistry为标准map容器，储存函数指针
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  //创建CreatorRegistry类型容器函数，返回其引用
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.// 向CreatorRegistry容器中增加Creator；
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry(); // 定义一个map的引用，通过这个函数创建一个
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.通过SolverParameter返回Solver指针；
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) //caffe.cpp 里的train函数会掉用这个的
  {
    LOG(INFO) <<"function Solver<Dtype>* CreateSolver() "<<" "<< "lijianfei debug!!!!!!!!!!";
    const string& type = param.type();  // 通常的SGD
    LOG(INFO) <<type<<" "<< "lijianfei debug!!!!!!!!!!";

    CreatorRegistry& registry = Registry();//通过调用Registry（）函数，Registry()中创建CreatorRegistry类的对象，定义了一个key类型为string，value类型为Creator的map：registry.其中Creator是一个solver函数指针类型，指向的函数的参数为SolverParameter类型
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";//果是一个已经register过的Solver类型，那么registry.count(type)应该为1
    return registry[type](param);//返回registry中type对应的creator对象，并调用这个creator函数，将creator返回的Solver<Dtype>*返回
  }

   //获取CreatorRegistry容器中注册过的solver类型名，string列表储存
  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry() {}

  //这个函数从solver_types列表中取出一个个string
  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};


template <typename Dtype>
class SolverRegisterer {
 public:
     //对SolverRegistry接口进行封装，功能是注册creator
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};


//注册方法一：注册一个solver creator
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \


//注册方法二
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
