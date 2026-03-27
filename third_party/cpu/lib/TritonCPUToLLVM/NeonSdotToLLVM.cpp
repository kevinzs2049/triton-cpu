/*
 * Lower NeonSdotOp → LLVM SDOT intrinsic
 * Lower CpuSdotGemvOp → runtime call to sdot_gemv_m1_prepacked()
 */

#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

// ---------- NeonSdotOp → LLVM intrinsic ----------

struct NeonSdotOpLowering : public OpRewritePattern<NeonSdotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NeonSdotOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto v4i32Ty = VectorType::get({4}, rewriter.getI32Type());
    auto sdotName = StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");
    auto result = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, v4i32Ty, sdotName,
        ValueRange{op.getAcc(), op.getA(), op.getB()},
        LLVM::FastmathFlagsAttr());
    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

// ---------- CpuSdotGemvOp → runtime function call ----------

struct SdotGemvOpLowering : public OpRewritePattern<triton::CpuSdotGemvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuSdotGemvOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();

    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Get or declare the runtime function
    auto funcName = "sdot_gemv_m1_prepacked";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    // Convert tt.ptr to llvm.ptr via unrealized_conversion_cast
    auto castToLLVMPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto aPtr = castToLLVMPtr(op.getAPtr());
    auto bPtr = castToLLVMPtr(op.getBPackedPtr());
    auto cPtr = castToLLVMPtr(op.getCPtr());

    auto four = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(4));
    auto N4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getN(), four);

    rewriter.create<LLVM::CallOp>(
        loc, funcOp, ValueRange{aPtr, bPtr, cPtr, op.getK(), op.getN(), N4});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotGemvFusedBf16Op → runtime call ----------

struct SdotGemvFusedBf16OpLowering
    : public OpRewritePattern<triton::CpuSdotGemvFusedBf16Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuSdotGemvFusedBf16Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sdot_gemv_m1_fused_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // 7 args: x_bf16, B_packed, w_scale, out_bf16, K, N, N4
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    // N4 = N / 4
    auto four = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(4));
    auto N4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getN(), four);

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getBPackedPtr()),
                   castPtr(op.getWScalePtr()), castPtr(op.getOutPtr()),
                   op.getK(), op.getN(), N4});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotPackWeightsOp → runtime call ----------

struct SdotPackWeightsOpLowering
    : public OpRewritePattern<triton::CpuSdotPackWeightsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuSdotPackWeightsOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sdot_pack_weights";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getBPtr()), castPtr(op.getBPackedPtr()),
                   op.getK(), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- Pass ----------

namespace mlir::triton::cpu {

std::unique_ptr<Pass> createNeonSdotToLLVMPass() {
  struct NeonSdotToLLVMPass : public OperationPass<ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NeonSdotToLLVMPass)
    NeonSdotToLLVMPass()
        : OperationPass<ModuleOp>(TypeID::get<NeonSdotToLLVMPass>()) {}
    StringRef getName() const override { return "NeonSdotToLLVM"; }
    std::unique_ptr<Pass> clonePass() const override {
      return std::make_unique<NeonSdotToLLVMPass>();
    }
    StringRef getArgument() const override {
      return "convert-neon-sdot-to-llvm";
    }
    void runOnOperation() override {
      auto *ctx = &getContext();
      RewritePatternSet patterns(ctx);
      patterns.add<NeonSdotOpLowering>(ctx);
      patterns.add<SdotGemvOpLowering>(ctx);
      patterns.add<SdotGemvFusedBf16OpLowering>(ctx);
      patterns.add<SdotPackWeightsOpLowering>(ctx);
      if (failed(applyPatternsGreedily(getOperation(),
                                               std::move(patterns))))
        signalPassFailure();
    }
  };
  return std::make_unique<NeonSdotToLLVMPass>();
}

} // namespace mlir::triton::cpu
