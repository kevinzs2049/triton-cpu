#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTOSVE2I8MM
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct SVE2I8MMDotOpCandidate {
  cpu::DotOp op;
};

bool isSVE2I8MMCandidate(cpu::DotOp op, SVE2I8MMDotOpCandidate &candidate) {
  auto aTy = dyn_cast<VectorType>(op.getA().getType());
  auto bTy = dyn_cast<VectorType>(op.getB().getType());
  auto cTy = dyn_cast<VectorType>(op.getC().getType());
  if (!aTy || !bTy || !cTy) {
    LDBG("Drop candidate with non-vector types.");
    return false;
  }

  if (aTy.getRank() != bTy.getRank() || aTy.getRank() != cTy.getRank() ||
      (aTy.getRank() != 2 && aTy.getRank() != 3)) {
    LDBG("Drop candidate with unsupported rank.");
    return false;
  }

  auto aElemTy = aTy.getElementType();
  auto bElemTy = bTy.getElementType();
  auto cElemTy = cTy.getElementType();

  if (!aElemTy.isInteger(8) || !bElemTy.isInteger(8)) {
    LDBG("Drop candidate with non-i8 inputs.");
    return false;
  }

  if (!cElemTy.isInteger(32)) {
    LDBG("Drop candidate with non-i32 accumulator.");
    return false;
  }

  candidate.op = op;
  return true;
}

LogicalResult convertCandidate(SVE2I8MMDotOpCandidate &candidate,
                               PatternRewriter &rewriter) {
  LDBG("Enter convertCandidate for SVE2 i8mm");
  // TODO: Implement SVE2 i8mm lowering (smmla/ummla) here.
  return rewriter.notifyMatchFailure(candidate.op,
                                     "TODO: implement SVE2 i8mm lowering");
}

struct ConvertDotToSVE2I8MM
    : public triton::cpu::impl::ConvertDotToSVE2I8MMBase<
          ConvertDotToSVE2I8MM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<SVE2I8MMDotOpCandidate> candidates;
    mod.walk([&candidates](cpu::DotOp op) {
      SVE2I8MMDotOpCandidate candidate;
      if (isSVE2I8MMCandidate(op, candidate)) {
        LLVM_DEBUG(LDBG("Found SVE2 i8mm candidate: " << op));
        candidates.push_back(candidate);
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(convertCandidate(candidate, rewriter))) {
        LDBG("SVE2 i8mm conversion succeeded.");
      } else {
        LDBG("SVE2 i8mm conversion skipped.");
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToSVE2I8MM() {
  return std::make_unique<ConvertDotToSVE2I8MM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
