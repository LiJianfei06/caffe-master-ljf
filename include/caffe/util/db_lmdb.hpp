#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {
// 下面的MDB_SUCCESS为一个宏定义,为0,表示成功,如果失败则对应不同的数值,表示不同的错误
// mdb_strerror,输出string,它的作用是根据不同的错误输出不同的错误语句
inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor) //  初始化
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);                  // mdb_cursor_close函数的作用为:关闭一个cursor句柄
    mdb_txn_abort(mdb_txn_);                        // 该函数的作用为:用于放弃所有对transaction的操作,并释放掉transaction句柄
  }
  virtual void SeekToFirst() { Seek(MDB_FIRST);}    // 把database里的第一个key/value的值放入变量:mdb_key_与mdb_value_
  virtual void Next() { Seek(MDB_NEXT); }           // 下一个key/value
  virtual string key() { // 返回mdb_key_里的数据,以字符串形式
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  virtual string value() { // 返回mdb_value_里的数据(以字符串的方式;,应该是用了strin的构造函数
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

 private:
  void Seek(MDB_cursor_op op) { // 注意,这里的MDB_cursor_op为枚举类型,代表了curcor的相关操作
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);   // 这个函数用于通过curcor恢复key与data
    if (mdb_status == MDB_NOTFOUND) {   // 当返回的状态为MDB_NOTFOUND时,表明,没有发现匹配的key
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  MDB_txn* mdb_txn_;            // 初始化会赋值的
  MDB_cursor* mdb_cursor_;      // 初始化会赋值的
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
};

class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_env* mdb_env)    // 给一个环境handle赋值
    : mdb_env_(mdb_env) { }
  virtual void Put(const string& key, const string& value); // 把key与value的值分别push到对应的vector里
  virtual void Commit(); // 它做的时,把keys 与 values里的数据提交 ,并清空它们

 private:
  MDB_env* mdb_env_;            // 环境句柄
  vector<string> keys, values;  // 两个vector容器

  void DoubleMapSize();         // 把环境的mapsize扩大两倍

  DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode); // 它所做的事情就是创建一个操作环境,根据mode,来决定是读还是NEW
  virtual void Close() {    // 它所做的就是:当所创建的环境的handle 不为空时,说明还没有释放掉
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_); // 于是呢,把相关的如database的handle,以及mdb_env_释放掉,释放先前的内存;
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual LMDBCursor* NewCursor();      // 根据mdb_env_,mdb_dbi_,创建了一个LMDBCursor的类
  virtual LMDBTransaction* NewTransaction();    // 返回一个用mdb_env_初始化了的LMDBTransaction类的指针

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
