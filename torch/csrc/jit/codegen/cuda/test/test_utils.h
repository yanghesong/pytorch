#pragma once

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <cstddef>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

namespace {
// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeSymbolicTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).build();
}

// Make a non-contiguous tensor of compile-time known sizes
TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

TensorView* makeContigConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .shape(shape)
      .dtype(dtype)
      .contiguity(std::vector<bool>(shape.size(), true))
      .build();
}

void checkIntValue(
    ExpressionEvaluator& evaluator,
    Val* val,
    Int::ScalarType expected_value) {
  TORCH_CHECK(val->isAnInt());
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

void checkIntValue(
    kir::ExpressionEvaluator& evaluator,
    const Val* val,
    Int::ScalarType expected_value) {
  const auto actual_value = evaluator.evaluate(val);
  TORCH_CHECK(actual_value.has_value());
  TORCH_CHECK(actual_value.value() == expected_value);
}

bool deviceMajorMinorCheck(int major, int minor = 0) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  if (dev_prop->major < major ||
      (dev_prop->major == major && dev_prop->minor < minor)) {
    return false;
  }
  return true;
}

int deviceSMCount() {
  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return sm_count;
}

void clearL2Cache() {
  torch::NoGradGuard no_grad;
  auto l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA, 0);

  auto l2_elems = l2_cache_size / 4;
  torch::Tensor t0 = torch::empty(l2_elems, options);
  torch::Tensor t1 = torch::clone(t0);
};

TensorView* loweredTv(TensorView* tv, GpuLower& gpulw) {
  auto used_tvs = ir_utils::allTvs(gpulw.kernel()->as<Fusion>());
  TensorView* matching_tv = nullptr;
  for (auto lowered_tv : used_tvs) {
    if (lowered_tv->name() == tv->name()) {
      matching_tv = lowered_tv;
    }
  }
  TORCH_INTERNAL_ASSERT(matching_tv != nullptr);
  return matching_tv;
}

class PredicatedChecker : public kir::IrVisitor {
 public:
  // Checks if the provided tv is written to within a non-trivial conditional
  static bool isPredicated(TensorView* tv, GpuLower& gpulw) {
    PredicatedChecker checker(
        loweredTv(tv, gpulw), gpulw.kernel()->topLevelExprs());
    return checker.is_predicated_;
  }

 private:
  PredicatedChecker() = delete;

  PredicatedChecker(TensorView* tv, std::vector<Expr*> exprs) : tv_(tv) {
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;
  bool is_predicated_ = false;
  bool predicated_ite_ = false;
  TensorView* tv_ = nullptr;

  void handle(kir::IfThenElse* ite) final {
    auto prev_ite = predicated_ite_;
    predicated_ite_ = !ite->predicate()->value()->isConstScalar();
    kir::IrVisitor::handle(ite);
    predicated_ite_ = prev_ite;
  }

  void handle(Expr* expr) final {
    if (expr->outputs().size() && expr->outputs()[0]->isA<kir::TensorIndex>()) {
      auto ti = expr->outputs()[0]->as<kir::TensorIndex>();
      if (ti->view() == tv_) {
        is_predicated_ = is_predicated_ | predicated_ite_;
        if (expr->predicate() != nullptr &&
            !expr->predicate()->value()->isConst()) {
          is_predicated_ = true;
        }
      }
    }
    kir::IrVisitor::handle(expr);
  }
};

class UnswitchInElseChecker : public kir::IrVisitor {
 public:
  // Checks if there are any unswitched for loops within an else clause
  static bool check(GpuLower& gpulw) {
    UnswitchInElseChecker checker(gpulw.kernel()->topLevelExprs());
    return checker.found_in_else_;
  }

 private:
  UnswitchInElseChecker() = delete;
  UnswitchInElseChecker(std::vector<Expr*> exprs) {
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;
  bool within_else_ = false;
  bool found_in_else_ = false;

  void handle(kir::IfThenElse* ite) final {
    auto prev_within_else = within_else_;
    within_else_ = true;
    kir::IrVisitor::handle(ite->elseBody().exprs());
    within_else_ = prev_within_else;
  }

  void handle(kir::ForLoop* for_loop) final {
    if (for_loop->iter_domain()->getParallelType() == ParallelType::Unswitch) {
      found_in_else_ = found_in_else_ || within_else_;
    }
    kir::IrVisitor::handle(for_loop);
  }
};

class PredicateMagicZeroChecker : public kir::IrVisitor {
 public:
  // Checks if all predicated domains of the provided tv are protected with
  // magic zero
  static bool isProtected(TensorView* tv, GpuLower& gpulw) {
    PredicateMagicZeroChecker checker(
        loweredTv(tv, gpulw), gpulw.kernel()->topLevelExprs());
    return checker.is_protected_;
  }

 private:
  using kir::IrVisitor::handle;

  PredicateMagicZeroChecker(TensorView* tv, std::vector<Expr*> exprs)
      : tv_(tv) {
    handle(exprs);
  }

  void handle(kir::IfThenElse* ite) final {
    auto prev_predicate = predicate_;
    predicate_ = ite->predicate()->value();
    kir::IrVisitor::handle(ite);
    predicate_ = prev_predicate;
  }

  void handle(Expr* expr) final {
    if (expr->outputs().size() && expr->outputs()[0]->isA<kir::TensorIndex>()) {
      auto ti = expr->outputs()[0]->as<kir::TensorIndex>();
      if (ti->view() == tv_) {
        is_protected_ = checkPredicateOfTensor(predicate_);
        return;
      }
    }

    if (expr->isA<kir::ForLoop>()) {
      handle(expr->as<kir::ForLoop>());
    } else if (expr->isA<kir::IfThenElse>()) {
      handle(expr->as<kir::IfThenElse>());
    } else {
      for (auto input : expr->inputs()) {
        handle(input);
      }
    }
  }

  // Return true If all predicated domains are protected
  bool checkPredicateOfTensor(Val* predicate) {
    auto id_predicates = decomposeCompoundPredicate(predicate);
    for (auto id_predicate : id_predicates) {
      // Just check if nvfuser_zero is used. Not perfect but probably
      // good enough.
      is_magic_zero_found_ = false;
      handle(id_predicate);
      if (!is_magic_zero_found_) {
        return false;
      }
    }
    return true;
  }

  // Decompose "X && Y" to a vector of {X, Y}.
  std::vector<Val*> decomposeCompoundPredicate(Val* predicate) {
    if (auto binary_op = dynamic_cast<BinaryOp*>(predicate->definition())) {
      if (binary_op->getBinaryOpType() == BinaryOpType::And) {
        auto pred = decomposeCompoundPredicate(binary_op->lhs());
        auto rhs_pred = decomposeCompoundPredicate(binary_op->rhs());
        pred.insert(pred.end(), rhs_pred.begin(), rhs_pred.end());
        return pred;
      }
    }

    return {predicate};
  }

  void handle(Val* val) final {
    if (isMagicZero(val)) {
      is_magic_zero_found_ = true;
      return;
    }

    auto def = val->definition();
    if (def != nullptr) {
      handle(def);
    }
  }

 private:
  bool is_protected_ = false;
  Val* predicate_ = nullptr;
  TensorView* tv_ = nullptr;
  bool is_magic_zero_found_ = false;
};

// Basically just TransformPropagator, except that it checks the consistency
// replayPasC with getMatchedLeafPosWithoutReplayPasC, replayCasP with
// getMatchedLeafPosWithoutReplayCasP, and fullSelfReplay with fullSelfMatching:
// - After replayPasC, getMatchedLeafPosWithoutReplayPasC should return the same
//   replayed position
// - After replayCasP, getMatchedLeafPosWithoutReplayCasP should return the same
//   replayed position
// - After fullSelfReplay, fullSelfMatching should return true
struct TransformPropagatorWithCheck : public TransformPropagator {
 public:
  virtual void propagateC2P(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateC2P(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayPasC(
            to, from, from_pos) == (int)to_pos);
  }
  virtual void propagateP2C(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateP2C(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayCasP(
            to, from, from_pos) == (int)to_pos);
  }
  virtual void propagateSibling(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateSibling(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(from_pos == to_pos);
    TORCH_CHECK(TransformReplay::fullSelfMatching(from, to));
  }
  using TransformPropagator::TransformPropagator;
};

} // namespace

class ContextCudnnTF32Disabled {
 public:
  ContextCudnnTF32Disabled() {
    flag_ = at::globalContext().allowTF32CuDNN();
    at::globalContext().setAllowTF32CuDNN(false);
  }

  ~ContextCudnnTF32Disabled() {
    at::globalContext().setAllowTF32CuDNN(flag_);
  }

 private:
  bool flag_;
};

// Fixture class must be uniquely identified, i.e., can't be in an
// anonymous namespace
class NVFuserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // requires PASCAL or newer
    if (!deviceMajorMinorCheck(6)) {
      GTEST_SKIP() << "skipping tests on pre-PASCAL GPUs";
    }
    setFillAllocationWithNan(true);
  }
};

} // namespace jit
} // namespace torch
