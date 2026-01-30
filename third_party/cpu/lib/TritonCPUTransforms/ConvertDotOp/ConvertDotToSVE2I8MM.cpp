#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

static Value cstI64(Location loc, int64_t v, PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, v, 64);
}

static Value extractElem2D(Location loc, Value vec, int64_t r, int64_t c,
                           PatternRewriter &rewriter) {
  Value row = rewriter.create<vector::ExtractOp>(loc, vec, r);
  return rewriter.create<vector::ExtractElementOp>(loc, row,
                                                   cstI64(loc, c, rewriter));
}

static Value packVec8ToNxv16(Location loc, Value vec8, Type nxv16i8Ty,
                             PatternRewriter &rewriter) {
  Value acc = rewriter.create<LLVM::UndefOp>(loc, nxv16i8Ty);
  for (int64_t idx = 0; idx < 8; ++idx) {
    Value elem = rewriter.create<vector::ExtractElementOp>(
        loc, vec8, cstI64(loc, idx, rewriter));
    acc = rewriter.create<LLVM::InsertElementOp>(
        loc, nxv16i8Ty, acc, elem, cstI64(loc, idx, rewriter));
  }
  return acc;
}

static Value packVec16ToNxv16(Location loc, Value vec16, Type nxv16i8Ty,
                              PatternRewriter &rewriter) {
  Value acc = rewriter.create<LLVM::UndefOp>(loc, nxv16i8Ty);
  for (int64_t idx = 0; idx < 16; ++idx) {
    Value elem = rewriter.create<vector::ExtractElementOp>(
        loc, vec16, cstI64(loc, idx, rewriter));
    acc = rewriter.create<LLVM::InsertElementOp>(
        loc, nxv16i8Ty, acc, elem, cstI64(loc, idx, rewriter));
  }
  return acc;
}

static Value zip1I64(Location loc, Value a, Value b, Type nxv16i8Ty,
                     PatternRewriter &rewriter) {
  Type nxv2i64Ty =
      LLVM::LLVMScalableVectorType::get(rewriter.getI64Type(), 2);
  Value a64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, a);
  Value b64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, b);
  StringAttr zip1 =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.sve.zip1.nxv2i64");
  Value z = rewriter
                .create<LLVM::CallIntrinsicOp>(loc, nxv2i64Ty, zip1,
                                               ValueRange{a64, b64})
                .getResult(0);
  return rewriter.create<LLVM::BitcastOp>(loc, nxv16i8Ty, z);
}

static Value zip2I64(Location loc, Value a, Value b, Type nxv16i8Ty,
                     PatternRewriter &rewriter) {
  Type nxv2i64Ty =
      LLVM::LLVMScalableVectorType::get(rewriter.getI64Type(), 2);
  Value a64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, a);
  Value b64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, b);
  StringAttr zip2 =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.sve.zip2.nxv2i64");
  Value z = rewriter
                .create<LLVM::CallIntrinsicOp>(loc, nxv2i64Ty, zip2,
                                               ValueRange{a64, b64})
                .getResult(0);
  return rewriter.create<LLVM::BitcastOp>(loc, nxv16i8Ty, z);
}

static Value pack2x8i8ToNxv16(Location loc, Value tile, Type nxv16i8Ty,
                              PatternRewriter &rewriter) {
  // Pack rows as [row0(8), row1(8)] which matches zip1 on 64-bit lanes.
  Value row0 = rewriter.create<vector::ExtractOp>(loc, tile, 0);
  Value row1 = rewriter.create<vector::ExtractOp>(loc, tile, 1);
  Value v0 = packVec8ToNxv16(loc, row0, nxv16i8Ty, rewriter);
  Value v1 = packVec8ToNxv16(loc, row1, nxv16i8Ty, rewriter);
  return zip1I64(loc, v0, v1, nxv16i8Ty, rewriter);
}

static Value pack2x2i32ToNxv4(Location loc, Value tile, Type nxv4i32Ty,
                              PatternRewriter &rewriter) {
  auto i32Ty = cast<VectorType>(tile.getType()).getElementType();
  auto i64Ty = rewriter.getI64Type();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto v2i64Ty = VectorType::get({2}, i64Ty);
  auto nxv2i64Ty = LLVM::LLVMScalableVectorType::get(i64Ty, 2);

  Value flat = rewriter.create<vector::ShapeCastOp>(loc, v4i32Ty, tile);
  Value v2i64 = rewriter.create<vector::BitCastOp>(loc, v2i64Ty, flat);
  Value lo = rewriter.create<vector::ExtractElementOp>(
      loc, v2i64, cstI64(loc, 0, rewriter));
  Value hi = rewriter.create<vector::ExtractElementOp>(
      loc, v2i64, cstI64(loc, 1, rewriter));

  Value acc64 = rewriter.create<LLVM::UndefOp>(loc, nxv2i64Ty);
  acc64 = rewriter.create<LLVM::InsertElementOp>(
      loc, nxv2i64Ty, acc64, lo, cstI64(loc, 0, rewriter));
  acc64 = rewriter.create<LLVM::InsertElementOp>(
      loc, nxv2i64Ty, acc64, hi, cstI64(loc, 1, rewriter));
  return rewriter.create<LLVM::BitcastOp>(loc, nxv4i32Ty, acc64);
}

static Value unpackNxv4To2x2i32(Location loc, Value vec, Type tileTy,
                                PatternRewriter &rewriter) {
  auto i64Ty = rewriter.getI64Type();
  auto i32Ty = cast<VectorType>(tileTy).getElementType();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto v2i64Ty = VectorType::get({2}, i64Ty);
  auto nxv2i64Ty = LLVM::LLVMScalableVectorType::get(i64Ty, 2);

  Value v2i64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, vec);
  Value lo =
      rewriter.create<LLVM::ExtractElementOp>(loc, i64Ty, v2i64,
                                              cstI64(loc, 0, rewriter));
  Value hi =
      rewriter.create<LLVM::ExtractElementOp>(loc, i64Ty, v2i64,
                                              cstI64(loc, 1, rewriter));

  auto zeroAttr = rewriter.getZeroAttr(v2i64Ty);
  Value tmp64 = rewriter.create<arith::ConstantOp>(loc, v2i64Ty, zeroAttr);
  tmp64 = rewriter.create<vector::InsertElementOp>(loc, lo, tmp64,
                                                   cstI64(loc, 0, rewriter));
  tmp64 = rewriter.create<vector::InsertElementOp>(loc, hi, tmp64,
                                                   cstI64(loc, 1, rewriter));
  Value flat = rewriter.create<vector::BitCastOp>(loc, v4i32Ty, tmp64);
  return rewriter.create<vector::ShapeCastOp>(loc, tileTy, flat);
}

static Value build4x4FromAccVecs(Location loc, Value acc00, Value acc01,
                                 Value acc10, Value acc11, Type tileTy,
                                 PatternRewriter &rewriter) {
  auto elemTy = cast<VectorType>(tileTy).getElementType();
  auto tile2x2Ty = VectorType::get({2, 2}, elemTy);
  Value tile00 = unpackNxv4To2x2i32(loc, acc00, tile2x2Ty, rewriter);
  Value tile01 = unpackNxv4To2x2i32(loc, acc01, tile2x2Ty, rewriter);
  Value tile10 = unpackNxv4To2x2i32(loc, acc10, tile2x2Ty, rewriter);
  Value tile11 = unpackNxv4To2x2i32(loc, acc11, tile2x2Ty, rewriter);

  auto zeroAttr = rewriter.getZeroAttr(tileTy);
  Value tile = rewriter.create<arith::ConstantOp>(loc, tileTy, zeroAttr);
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile00, tile, ArrayRef<int64_t>{0, 0}, ArrayRef<int64_t>{1, 1});
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile01, tile, ArrayRef<int64_t>{0, 2}, ArrayRef<int64_t>{1, 1});
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile10, tile, ArrayRef<int64_t>{2, 0}, ArrayRef<int64_t>{1, 1});
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile11, tile, ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
  return tile;
}

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
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();

  auto aTy = cast<VectorType>(op.getA().getType());
  auto bTy = cast<VectorType>(op.getB().getType());
  auto cTy = cast<VectorType>(op.getC().getType());

  if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "rank != 2 not supported yet");

  int64_t M = aTy.getDimSize(0);
  int64_t K = aTy.getDimSize(1);
  int64_t N = bTy.getDimSize(1);

  if (bTy.getDimSize(0) != K || cTy.getDimSize(0) != M ||
      cTy.getDimSize(1) != N)
    return rewriter.notifyMatchFailure(op, "shape mismatch");

  // Fixed 128-bit SVE width micro-kernel: 2x2x8 or 2x2x16 (smmla).
  // We tile the outer loop as 4x4 to reduce loop overhead.
  if (M % 4 != 0 || N % 4 != 0 || K % 8 != 0)
    return rewriter.notifyMatchFailure(
        op, "requires M,N divisible by 4 and K divisible by 8");

  Type i8Ty = aTy.getElementType();
  Type i32Ty = cTy.getElementType();
  Type nxv16i8Ty = LLVM::LLVMScalableVectorType::get(i8Ty, 16);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);

  StringAttr smmla =
      StringAttr::get(op.getContext(), "llvm.aarch64.sve.smmla.nxv4i32");

  Value res = op.getC();
  for (int64_t n = 0; n < N; n += 4) {
    if (K % 16 == 0) {
      SmallVector<Value, 4> bLo0s;
      SmallVector<Value, 4> bHi0s;
      SmallVector<Value, 4> bLo1s;
      SmallVector<Value, 4> bHi1s;
      bLo0s.reserve(K / 16);
      bHi0s.reserve(K / 16);
      bLo1s.reserve(K / 16);
      bHi1s.reserve(K / 16);

      for (int64_t k = 0; k < K; k += 16) {
        Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, op.getB(), ArrayRef<int64_t>{k, n + 0},
            ArrayRef<int64_t>{16, 2}, ArrayRef<int64_t>{1, 1});
        Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, op.getB(), ArrayRef<int64_t>{k, n + 2},
            ArrayRef<int64_t>{16, 2}, ArrayRef<int64_t>{1, 1});

        Value bTile0T = rewriter.create<vector::TransposeOp>(
            loc, bTile0, ArrayRef<int64_t>{1, 0});
        Value bTile1T = rewriter.create<vector::TransposeOp>(
            loc, bTile1, ArrayRef<int64_t>{1, 0});

        Value bRow00 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 0);
        Value bRow01 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 1);
        Value bRow10 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 0);
        Value bRow11 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 1);
        Value bVec00 = packVec16ToNxv16(loc, bRow00, nxv16i8Ty, rewriter);
        Value bVec01 = packVec16ToNxv16(loc, bRow01, nxv16i8Ty, rewriter);
        Value bVec10 = packVec16ToNxv16(loc, bRow10, nxv16i8Ty, rewriter);
        Value bVec11 = packVec16ToNxv16(loc, bRow11, nxv16i8Ty, rewriter);
        Value bLo0 = zip1I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
        Value bHi0 = zip2I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
        Value bLo1 = zip1I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
        Value bHi1 = zip2I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);

        bLo0s.push_back(bLo0);
        bHi0s.push_back(bHi0);
        bLo1s.push_back(bLo1);
        bHi1s.push_back(bHi1);
      }

      for (int64_t m = 0; m < M; m += 4) {
        // Four 2x2 sub-tiles inside each 4x4 block.
        Value accTile00 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 0, n + 0}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});
        Value accTile01 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 0, n + 2}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});
        Value accTile10 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 2, n + 0}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});
        Value accTile11 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 2, n + 2}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});

        Value accVec00 = pack2x2i32ToNxv4(loc, accTile00, nxv4i32Ty, rewriter);
        Value accVec01 = pack2x2i32ToNxv4(loc, accTile01, nxv4i32Ty, rewriter);
        Value accVec10 = pack2x2i32ToNxv4(loc, accTile10, nxv4i32Ty, rewriter);
        Value accVec11 = pack2x2i32ToNxv4(loc, accTile11, nxv4i32Ty, rewriter);

        int64_t ki = 0;
        for (int64_t k = 0; k < K; k += 16, ++ki) {
          Value aTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, op.getA(), ArrayRef<int64_t>{m + 0, k},
              ArrayRef<int64_t>{2, 16}, ArrayRef<int64_t>{1, 1});
          Value aTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, op.getA(), ArrayRef<int64_t>{m + 2, k},
              ArrayRef<int64_t>{2, 16}, ArrayRef<int64_t>{1, 1});

          Value aRow00 = rewriter.create<vector::ExtractOp>(loc, aTile0, 0);
          Value aRow01 = rewriter.create<vector::ExtractOp>(loc, aTile0, 1);
          Value aRow10 = rewriter.create<vector::ExtractOp>(loc, aTile1, 0);
          Value aRow11 = rewriter.create<vector::ExtractOp>(loc, aTile1, 1);
          Value aVec00 = packVec16ToNxv16(loc, aRow00, nxv16i8Ty, rewriter);
          Value aVec01 = packVec16ToNxv16(loc, aRow01, nxv16i8Ty, rewriter);
          Value aVec10 = packVec16ToNxv16(loc, aRow10, nxv16i8Ty, rewriter);
          Value aVec11 = packVec16ToNxv16(loc, aRow11, nxv16i8Ty, rewriter);
          Value aLo0 = zip1I64(loc, aVec00, aVec01, nxv16i8Ty, rewriter);
          Value aHi0 = zip2I64(loc, aVec00, aVec01, nxv16i8Ty, rewriter);
          Value aLo1 = zip1I64(loc, aVec10, aVec11, nxv16i8Ty, rewriter);
          Value aHi1 = zip2I64(loc, aVec10, aVec11, nxv16i8Ty, rewriter);

          Value bLo0 = bLo0s[ki];
          Value bHi0 = bHi0s[ki];
          Value bLo1 = bLo1s[ki];
          Value bHi1 = bHi1s[ki];

          accVec00 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec00, aLo0, bLo0})
                         .getResult(0);
          accVec00 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec00, aHi0, bHi0})
                         .getResult(0);

          accVec01 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec01, aLo0, bLo1})
                         .getResult(0);
          accVec01 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec01, aHi0, bHi1})
                         .getResult(0);

          accVec10 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec10, aLo1, bLo0})
                         .getResult(0);
          accVec10 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec10, aHi1, bHi0})
                         .getResult(0);

          accVec11 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec11, aLo1, bLo1})
                         .getResult(0);
          accVec11 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec11, aHi1, bHi1})
                         .getResult(0);
        }

        auto tile4x4Ty = VectorType::get({4, 4}, i32Ty);
        Value tile4x4 = build4x4FromAccVecs(loc, accVec00, accVec01, accVec10,
                                            accVec11, tile4x4Ty, rewriter);
        res = rewriter.create<vector::InsertStridedSliceOp>(
            loc, tile4x4, res, ArrayRef<int64_t>{m, n}, ArrayRef<int64_t>{1, 1});
      }
    } else {
      SmallVector<Value, 4> bVec0s;
      SmallVector<Value, 4> bVec1s;
      bVec0s.reserve(K / 8);
      bVec1s.reserve(K / 8);

      for (int64_t k = 0; k < K; k += 8) {
        Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, op.getB(), ArrayRef<int64_t>{k, n + 0},
            ArrayRef<int64_t>{8, 2}, ArrayRef<int64_t>{1, 1});
        Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, op.getB(), ArrayRef<int64_t>{k, n + 2},
            ArrayRef<int64_t>{8, 2}, ArrayRef<int64_t>{1, 1});

        Value bTile0T = rewriter.create<vector::TransposeOp>(
            loc, bTile0, ArrayRef<int64_t>{1, 0});
        Value bTile1T = rewriter.create<vector::TransposeOp>(
            loc, bTile1, ArrayRef<int64_t>{1, 0});

        Value bVec0 = pack2x8i8ToNxv16(loc, bTile0T, nxv16i8Ty, rewriter);
        Value bVec1 = pack2x8i8ToNxv16(loc, bTile1T, nxv16i8Ty, rewriter);
        bVec0s.push_back(bVec0);
        bVec1s.push_back(bVec1);
      }

      for (int64_t m = 0; m < M; m += 4) {
        // Four 2x2 sub-tiles inside each 4x4 block.
        Value accTile00 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 0, n + 0}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});
        Value accTile01 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 0, n + 2}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});
        Value accTile10 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 2, n + 0}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});
        Value accTile11 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, res, ArrayRef<int64_t>{m + 2, n + 2}, ArrayRef<int64_t>{2, 2},
            ArrayRef<int64_t>{1, 1});

        Value accVec00 = pack2x2i32ToNxv4(loc, accTile00, nxv4i32Ty, rewriter);
        Value accVec01 = pack2x2i32ToNxv4(loc, accTile01, nxv4i32Ty, rewriter);
        Value accVec10 = pack2x2i32ToNxv4(loc, accTile10, nxv4i32Ty, rewriter);
        Value accVec11 = pack2x2i32ToNxv4(loc, accTile11, nxv4i32Ty, rewriter);

        int64_t ki = 0;
        for (int64_t k = 0; k < K; k += 8, ++ki) {
          Value aTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, op.getA(), ArrayRef<int64_t>{m + 0, k},
              ArrayRef<int64_t>{2, 8}, ArrayRef<int64_t>{1, 1});
          Value aTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, op.getA(), ArrayRef<int64_t>{m + 2, k},
              ArrayRef<int64_t>{2, 8}, ArrayRef<int64_t>{1, 1});

          Value aVec0 = pack2x8i8ToNxv16(loc, aTile0, nxv16i8Ty, rewriter);
          Value aVec1 = pack2x8i8ToNxv16(loc, aTile1, nxv16i8Ty, rewriter);
          Value bVec0 = bVec0s[ki];
          Value bVec1 = bVec1s[ki];

          accVec00 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec00, aVec0, bVec0})
                         .getResult(0);
          accVec01 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec01, aVec0, bVec1})
                         .getResult(0);
          accVec10 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec10, aVec1, bVec0})
                         .getResult(0);
          accVec11 = rewriter
                         .create<LLVM::CallIntrinsicOp>(
                             loc, nxv4i32Ty, smmla,
                             ValueRange{accVec11, aVec1, bVec1})
                         .getResult(0);
        }

        auto tile4x4Ty = VectorType::get({4, 4}, i32Ty);
        Value tile4x4 = build4x4FromAccVecs(loc, accVec00, accVec01, accVec10,
                                            accVec11, tile4x4Ty, rewriter);
        res = rewriter.create<vector::InsertStridedSliceOp>(
            loc, tile4x4, res, ArrayRef<int64_t>{m, n}, ArrayRef<int64_t>{1, 1});
      }
    }
  }

  rewriter.replaceOp(op, res);
  return success();
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
