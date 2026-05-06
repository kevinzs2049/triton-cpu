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

// ---------- CpuFusedMlpOp → runtime call ----------

struct FusedMlpOpLowering : public OpRewritePattern<triton::CpuFusedMlpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuFusedMlpOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "fused_mlp_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty},
          false);
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
        ValueRange{castPtr(op.getXPtr()),
                   castPtr(op.getGatePackedPtr()),
                   castPtr(op.getUpPackedPtr()),
                   castPtr(op.getGateScalePtr()),
                   castPtr(op.getUpScalePtr()),
                   castPtr(op.getOutPtr()),
                   op.getK(), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuFlashAttnDecodeOp → runtime call ----------

struct FlashAttnDecodeOpLowering
    : public OpRewritePattern<triton::CpuFlashAttnDecodeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuFlashAttnDecodeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "flash_attn_decode_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy,
                   i64Ty, i64Ty, f32Ty, i64Ty, i64Ty, i64Ty, i64Ty}, false);
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

    auto smScaleVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Ty, op.getSmScaleAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getQPtr()), castPtr(op.getKPtr()),
                   castPtr(op.getVPtr()), castPtr(op.getOutPtr()),
                   op.getSeqLen(), op.getHeadDim(), smScaleVal,
                   op.getNumHeads(), op.getNumKvHeads(),
                   op.getStrideKn(), op.getStrideVn()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuRmsNormGatedOp → runtime call ----------

struct RmsNormGatedOpLowering
    : public OpRewritePattern<triton::CpuRmsNormGatedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuRmsNormGatedOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "standalone_rms_norm_gated_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, f32Ty}, false);
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
    auto epsVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Ty, op.getEpsAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getGatePtr()),
                   castPtr(op.getWeightPtr()), castPtr(op.getOutPtr()),
                   op.getM(), op.getD(), epsVal});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuRmsNormOp → runtime call ----------

struct RmsNormOpLowering : public OpRewritePattern<triton::CpuRmsNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuRmsNormOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "standalone_rms_norm_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, f32Ty}, false);
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

    // Extract eps from F32Attr
    auto epsVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Ty, op.getEpsAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getWeightPtr()),
                   castPtr(op.getOutPtr()), op.getD(), epsVal});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuCausalConv1dUpdateOp → runtime call ----------

struct CausalConv1dUpdateOpLowering
    : public OpRewritePattern<triton::CpuCausalConv1dUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuCausalConv1dUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "standalone_causal_conv1d_update_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // 5 ptrs + 4 i64
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy,
          {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
           i64Ty, i64Ty, i64Ty, i64Ty, i64Ty},
          false);
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

    SmallVector<Value, 10> args;
    args.push_back(castPtr(op.getHiddenPtr()));
    args.push_back(castPtr(op.getStatePtr()));
    args.push_back(castPtr(op.getWeightPtr()));
    args.push_back(castPtr(op.getBiasPtr()));
    args.push_back(castPtr(op.getOutPtr()));
    args.push_back(op.getB());
    args.push_back(op.getC());
    args.push_back(op.getKernelSize());
    args.push_back(op.getSilu());
    args.push_back(op.getHasBias());
    rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuGatedDeltaDecodeOp → runtime call ----------

struct GatedDeltaDecodeOpLowering
    : public OpRewritePattern<triton::CpuGatedDeltaDecodeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuGatedDeltaDecodeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "standalone_gated_delta_decode_fp32";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // 7 ptrs + 5 i64
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy,
          {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
           i64Ty, i64Ty, i64Ty, i64Ty, i64Ty},
          false);
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

    SmallVector<Value, 12> args;
    args.push_back(castPtr(op.getQPtr()));
    args.push_back(castPtr(op.getKPtr()));
    args.push_back(castPtr(op.getVPtr()));
    args.push_back(castPtr(op.getGPtr()));
    args.push_back(castPtr(op.getBetaPtr()));
    args.push_back(castPtr(op.getStatePtr()));
    args.push_back(castPtr(op.getOutPtr()));
    args.push_back(op.getB());
    args.push_back(op.getH());
    args.push_back(op.getKDim());
    args.push_back(op.getVDim());
    args.push_back(op.getUseL2norm());
    rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSwigluOp → runtime call ----------

struct SwigluOpLowering : public OpRewritePattern<triton::CpuSwigluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::CpuSwigluOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "swiglu_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty}, false);
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
        ValueRange{castPtr(op.getGatePtr()), castPtr(op.getUpPtr()),
                   castPtr(op.getOutPtr()), op.getN()});
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
      patterns.add<RmsNormOpLowering>(ctx);
      patterns.add<RmsNormGatedOpLowering>(ctx);
      patterns.add<GatedDeltaDecodeOpLowering>(ctx);
      patterns.add<CausalConv1dUpdateOpLowering>(ctx);
      patterns.add<SwigluOpLowering>(ctx);
      patterns.add<FlashAttnDecodeOpLowering>(ctx);
      patterns.add<FusedMlpOpLowering>(ctx);
      if (failed(applyPatternsGreedily(getOperation(),
                                               std::move(patterns))))
        signalPassFailure();
    }
  };
  return std::make_unique<NeonSdotToLLVMPass>();
}

} // namespace mlir::triton::cpu
