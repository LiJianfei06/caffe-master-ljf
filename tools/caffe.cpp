/*每次我们启动脚本文件或者从命令行输入命令开始训练深度神经网络时，总是从这个文件开始对命令进行解析并执行*/

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>  // gflags是google的一个开源的处理命令行参数的库。
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

/*====把名字空间声明一下============*/
using caffe::Blob;  
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

/*gflags是google的一个开源的处理命令行参数的库。在使用命令行参数的文件文件中（源文件或头文件），首先使用一下定义语句进行变量的定义。DEFINE_int32，DEFINE_int64，DEFINE_bool，DEFINE_double，DEFINE_string等，语法为：DEFINE_int32(name, default_value, "description")。接着你就可以使用FLAGS_name变量了，这些变量的值则是由命令行参数传递，无则为默认值，在其他代码文件中若想用该命令参数，可以用DECLARE_int32(name)声明（name为int32类型，也可以使用其他支持的类型）。在caffe.cpp中有很多FLAGS_name定义，如DEFINE_string(gpu,"","some description"），则命令行后-gpu 0，表示FLAGS_gpu=0，默认值为空。*/
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");  // 设置gpu的id
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");    // 设置solve.prototxt的路径
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");         // 设置继续上次训练的模型路径
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");   // 设置预训练的模型路径
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();  // 这里定义函数指针类型BrewFunction
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;     // 一个map容器，first是字符串，second是函数指针

/*下面的#define RegisterBrewFunction(func)宏定义的作用是将参数func转化为字符串，并存储在g_brew_map 
 * 这个容器中，而func对应了四种值：train/test/time/device_query，这四个函数标志了四个功能的入口
 * 留意下方对应的那四个函数的尾部就可得知。*/
// 宏定义，比如RegisterBrewFunction（train）时，相当于在容器g_brew_map中注册了train函数的函数指针和其对应的名字“train”
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
    LOG(INFO) << "func:"<<#func<<"lijianfei debug!!!!!!!!!!!!"; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}
/*C++中#和##用法:在C/C++的宏中，”#”的功能是将其后面的宏参数进行字符串化操作(Stringfication)，简单说就是在对它所引用的宏变量通过替换后在其左右各加上一个双引号。”##”被称为连接符(concatenator)，用来将两个子串Token连接为一个Token。注意这里连接的对象是Token就行，而不一定是宏的变量。
凡是宏定义里有用’#’或’##’的地方宏参数是不会再展开。若要使’#’和’##’的宏参数被展开，可以加多一层中间转换宏。*/




static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];    // 若找到这个函数就返回函数指针
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

/*=======./=====*/
// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);    //  RegisterBrewFunction将此函数入口添加进了g_brew_map

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() 
{
  LOG(INFO) <<"function train() "<<" "<< "lijianfei debug!!!!!!!!!!";
    
    
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train."; // google的glog库，检查--solver、--snapshot和--weight并输出消息；必须有指定solver，snapshot和weight两者指定其一
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();



// 实例化SolverParameter类，该类保存solver参数和相应方法（SoverParameter是由google protobuffer编译过来的类，具体声明可以见代码文件build/src/caffe/proto/caffe.pb.h）
  caffe::SolverParameter solver_param;
  // 将-solver指定solver.prototxt文件内容解析到solver_param中，该函数声明在include/caffe/util/upgrade_proto.hpp中，实现在src/caffe/util/upgrade_proto.cpp中；
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);       // 这些个函数都定义在caffe.pb.h中
    LOG(INFO) << "stages:"<<stages[i]<<"lijianfei debug!!!!!!!!!!!!";
  }

  /*下面是去查询用户配置的GPU信息，用户可以在输入命令行的时候配置gpu信息，也可以在solver.prototxt 
   *   文件中定义GPU信息，如果用户在solver.prototxt里面配置了GPU的id，则将该id写入FLAGS_gpu中，如果用户 
   *     只是说明了使用gpu模式，而没有详细指定使用的gpu的id，则将gpu的id默认为0。*/ 
  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.has_solver_mode()
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  /*在以下部分核验gpu检测结果，如果没有gpu信息，那么则使用cpu训练，否则，就开始一些GPU训练的初始化工作*/
  // 多GPU下，将GPU编号存入vector容器中（get_gpus()函数通过FLAGS_gpu获取）；
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  // 处理snapshot, stop or none信号，其声明在include/caffe/util/signal_Handler.h中；
  //   // GetRequestedAction在caffe.cpp中，将‘stop’，‘snapshot’，‘none’转换为标准信号，即解析；
  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  if (FLAGS_snapshot.size()) {
    solver_param.clear_weights();
  } else if (FLAGS_weights.size()) {
    solver_param.clear_weights();
    solver_param.add_weights(FLAGS_weights);
  }


  /*下面就开始构造网络训练器solver，调用SolverRegistry的CreateSolver函数得到一个solver，在初始化solver的过程中， 
   *   使用了之前解析好的用户定义的solver.prototxt文件，solver负担了整个网络的训练责任*/  
  // 声明boost库中智能指针solver，指向caffe::Solver对象，该对象由CreateSolver创建
  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param)); // 定义在solver_factory.hpp 中
  LOG(INFO) << "shared_ptr<caffe::Solver<float> >  finish!!   lijianfei debug!!!!!!!!!!";

  // Solver对象中方法的使用
  solver->SetActionFunction(signal_handler.GetActionFunction());


  /*在这里查询了一下用户有没有定义snapshot参数和weights参数，因为如果定义了这两个参数，代表用户可能会希望从之前的 
   *   中断训练处继续训练或者借用其他模型初始化网络，caffe在对两个参数相关的内容进行处理时都要用到solver指针*/  
  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  }

  LOG(INFO) << "Starting Optimization";
  if (gpus.size() > 1) /*如果有不止一块gpu参与训练，那么将开启多gpu训练模式*/ 
  {
#ifdef USE_NCCL
    caffe::NCCL<float> nccl(solver);
    nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
    LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
  } 
  else
  {
      /*使用Solve()接口正式开始优化网络*/  
    solver->Solve();// 初始化完成，开始优化网络（核心，重要）,接下来，CreateSolver函数和Solver类是需要弄清楚的。
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);    // RegisterBrewFunction将此函数入口添加进了g_brew_map


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);     // RegisterBrewFunction将此函数入口添加进了g_brew_map


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);     // RegisterBrewFunction将此函数入口添加进了g_brew_map

int main(int argc, char** argv) 
{
  int test_temp=0;
  /*google glog test：-----------
   LOG(INFO)其实返回的是一个std::ostream，对于不同级别的日志，返回的流是不一样的。日志级别有DUBUG，INFO，ERROR，FATAL。--  */
  LOG(INFO) << "google glog test!"<<" lijianfei debug!!!!!!!!!!";       // 输出信息，前面能显示行号(第一个字母I)
  LOG(ERROR) << "LOG(ERROR) test!"<<" lijianfei debug!!!!!!!!!!";       // 输出信息，前面能显示行号(第一个字母E)
  //LOG(FATAL) << "LOG(FATAL) test!"<<" lijianfei debug!!!!!!!!!!";       // 输出信息，前面能显示行号(第一个字母F)且会崩
  test_temp=2;
  //CHECK_EQ(test_temp, 2);   // 不相等则崩
  //CHECK_GE(test_temp,2);        // 不是大于的就崩
  //CHECK_LE(test_temp，2);        // 
  //CHECK_GT(test_temp,2);        //


  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;    // gflags库，具体说明紧接代码（未找到其定义，估计在gflags库文件中定义）
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  LOG(INFO) << "argc:"<<argc<<" lijianfei debug!!!!!!!!!!";       // 训练为例: 3 
  LOG(INFO) << "argv[0]:"<<argv[0]<<" lijianfei debug!!!!!!!!!!"; // 训练为例：./build/tools/caffe 
  LOG(INFO) << "argv[1]:"<<argv[1]<<" lijianfei debug!!!!!!!!!!"; // 训练为例：train 
  LOG(INFO) << "argv[2]:"<<argv[2]<<" lijianfei debug!!!!!!!!!!"; // 训练为例：--solver=examples/mnist/lenet_solver.prototxt 


  /*下面进行的是对gflags和glog的一些初始化，GlobalInit函数定义在了caffe安装目录./src/caffe/common.cpp中， 
    在下面贴出该函数的代码 
    void GlobalInit(int* pargc, char*** pargv) 
    { 
  // Google flags. 
  ::gflags::ParseCommandLineFlags(pargc, pargv, true); 
  // Google logging. 
  ::google::InitGoogleLogging(*(pargv)[0]); 
  // Provide a backtrace on segfault. 
  ::google::InstallFailureSignalHandler(); 
  }在该函数中，ParseCommandLineFlags函数对gflags的参数进行了初始化，InitGoogleLogging函数初始化谷歌日志系统， 
  而InstallFailureSignalHandler注册信号处理句柄*/  
  caffe::GlobalInit(&argc, &argv);  // include/caffe/commom.hpp中声明的函数：Currently it initializes google flags and google logging.即初始化FLAGS.
  // 判断参数，参数为2，继续执行action函数，否则输出usage信息。
  
  if (argc == 2) 
  {
#ifdef WITH_PYTHON_LAYER
    try {
      LOG(INFO) <<"use WITH_PYTHON_LAYER"<<" lijianfei debug!!!!!!!!!!";       // 训练为例: 3 
#endif
      /*上面完成了一些初始化工作，而真正的程序入口就是下面这个GetBrewFunction函数，这个函数的主要功能为去查找g_brew_map容器， 
       *   并在其中找到与caffe::string(argv[1])相匹配的函数并返回该函数的入口，那么，g_brew_map容器里面装的是什么呢？这个时候就要 
       *     看看上面的#define RegisterBrewFunction(func)。*/  
        /*在看完#define RegisterBrewFunction(func)之后，我们转向上文阅读一下GetBrewFunction的定义*/
      return GetBrewFunction(caffe::string(argv[1]))(); // GetBrewFunction函数返回函数指针，对于上面标准指令，则返回train函数指针     .那么就执行函数了
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
