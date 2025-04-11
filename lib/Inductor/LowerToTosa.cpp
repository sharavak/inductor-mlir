#include "Inductor/InductorDialect.h"
#include "Inductor/InductorOps.h"
#include "Inductor/InductorPasses.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"


#include "Utils/ShapeType.h"
#include<iostream>


#include <iostream>
#include <utility>
using namespace std;

class AddOpLowering : public mlir::OpRewritePattern<inductor::AddOp> {
  using OpRewritePattern<inductor::AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(inductor::AddOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type resultType = op.getResult().getType();
    mlir::tosa::AddOp addop = rewriter.create<mlir::tosa::AddOp>(
        op.getLoc(), resultType, op.getOperands());
    rewriter.replaceOp(op, addop);

    return mlir::success();
  }
};

class BatchNorm2dOpLowering
    : public mlir::OpRewritePattern<inductor::BatchNorm2dOp> {

  using OpRewritePattern<inductor::BatchNorm2dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(inductor::BatchNorm2dOp op,
                  mlir::PatternRewriter &rewriter) const override {

    // General Formula -> (x-mean)/sqrt(variance+eps)
    // Variance Formula -> (x-mean)*(x-mean)/N
    // Average(Mean) Formula -> (sum of all elements)/N

    mlir::Location loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = input.getType();
    auto inputTensorType = inputType.cast<mlir::RankedTensorType>();
    auto inputShape = inputTensorType.getShape();
    auto affine = op.getAffine();

    mlir::SmallVector<int64_t> shape(4, 0);
    shape = {1, inputShape[1], inputShape[2], inputShape[3]};
    // ---- For getting avg sum for the axis 0 2 3

    auto ip = mlir::RankedTensorType::get(
        shape, inputType.getElementType()); // (1,C,W,H)
    mlir::Value btsum = rewriter.create<mlir::tosa::ReduceSumOp>(
        loc, ip, input, rewriter.getI32IntegerAttr({0}));
    shape = {1, ip.getShape()[1], 1, ip.getShape()[3]};

    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value wisum = rewriter.create<mlir::tosa::ReduceSumOp>(
        loc, ip, btsum, rewriter.getI32IntegerAttr({2})); // (1,C,1,H)
    shape = {1, ip.getShape()[1], 1, 1};

    ip = mlir::RankedTensorType::get(shape,
                                     inputType.getElementType()); // (1,C,1,1)
    mlir::Value hisum = rewriter.create<mlir::tosa::ReduceSumOp>(
        loc, ip, wisum, rewriter.getI32IntegerAttr({3})); // batch sum (1,C,1,1)

    float val = 1.0 / (inputShape[0] * inputShape[2] * inputShape[3]);
    auto const_type = mlir::RankedTensorType::get({1, inputShape[1], 1, 1},
                                                  rewriter.getF32Type());
    auto const_attr = mlir::DenseElementsAttr::get(const_type, val);

    auto divisor = rewriter.create<mlir::tosa::ConstOp>(
        loc, const_type, const_attr); // (1,C,1,1)

    auto shiftType =
        mlir::RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto shiftAttr = mlir::DenseElementsAttr::get(
        shiftType, rewriter.getIntegerAttr(rewriter.getIntegerType(8), 1));
    auto constShift = rewriter.create<mlir::tosa::ConstOp>(
        op->getLoc(), shiftType, shiftAttr);

    auto width_mean = rewriter.create<mlir::tosa::MulOp>(
        loc, ip, hisum, divisor, constShift.getResult()); // batch avg (1,C,1,1)

    // Variance sum for the axis 0 2 3
    // (X-mean)
    auto var_sub =
        rewriter.create<mlir::tosa::SubOp>(loc, inputType, input, width_mean);
    // {b,h,w,h}

    shape = {1, inputShape[1], inputShape[2], inputShape[3]};
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    auto btVarSum = rewriter.create<mlir::tosa::ReduceSumOp>(
        loc, ip, var_sub, rewriter.getI32IntegerAttr({0}));

    shape = {1, ip.getShape()[1], 1, ip.getShape()[3]};
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    auto wiVarSum = rewriter.create<mlir::tosa::ReduceSumOp>(
        loc, ip, btVarSum, rewriter.getI32IntegerAttr({2}));

    shape = {1, ip.getShape()[1], 1, 1};
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value bt_sum = rewriter.create<mlir::tosa::ReduceSumOp>(
        loc, ip, wiVarSum, rewriter.getI32IntegerAttr({3})); // variance sum

    auto batch_var = rewriter.create<mlir::tosa::MulOp>(
        loc, inputType, bt_sum, divisor, constShift.getResult()); // variance

    float eps = 1e-5;
    auto epsType =
        mlir::RankedTensorType::get({1, 1, 1, 1}, rewriter.getF32Type());
    auto epsAttr = mlir::DenseElementsAttr::get(
        epsType, rewriter.getFloatAttr(rewriter.getF32Type(), eps));
    auto epsinter =
        rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), epsType, epsAttr);

    auto varAdd = rewriter.create<mlir::tosa::AddOp>(
        loc, inputType, batch_var, epsinter); // (variance+eps)

    auto rec_sqrt = rewriter.create<mlir::tosa::RsqrtOp>(
        loc, inputType, varAdd); // 1/(sqrt(variance+eps))

    auto xnorm = rewriter.create<mlir::tosa::MulOp>(
        loc, inputType, bt_sum, rec_sqrt, constShift.getResult());
    if (affine) {
      // #x_norm = x_norm*gamma + beta
      auto gammaType = mlir::RankedTensorType::get({1, inputShape[1], 1, 1},
                                                   rewriter.getF32Type());
      auto gammaAttr = mlir::DenseElementsAttr::get(
          gammaType, rewriter.getFloatAttr(rewriter.getF32Type(), 1.0));
      auto gamma = rewriter.create<mlir::tosa::ConstOp>(loc, gammaType,
                                                        gammaAttr); // (1,C,1,1)
      xnorm = rewriter.create<mlir::tosa::MulOp>(loc, inputType, xnorm, gamma,
                                                 constShift.getResult());

      auto betaType = mlir::RankedTensorType::get({1, inputShape[1], 1, 1},
                                                  rewriter.getF32Type());
      auto betaAttr = mlir::DenseElementsAttr::get(
          betaType, rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));

      auto beta = rewriter.create<mlir::tosa::ConstOp>(loc, betaType,
                                                       betaAttr); // (1,C,1,1)

      auto new_xnorm =
          rewriter.create<mlir::tosa::AddOp>(loc, inputType, xnorm, beta);
      rewriter.replaceOp(op, new_xnorm);
    } else {
      rewriter.replaceOp(op, xnorm);
    }

    return mlir::success();
  }
};

class ProdLowering : public mlir::OpRewritePattern<inductor::ProdOp> {
  using OpRewritePattern<inductor::ProdOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(inductor::ProdOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = input.getType();
    auto inputTensorType = inputType.cast<mlir::RankedTensorType>();
    auto inputShape = inputTensorType.getShape();

    auto keepdim = op.getKeepdim();
    mlir::SmallVector<int64_t> newShape;
    if (llvm::isa<mlir::IntegerAttr>(op.getDimAttr())) {
      auto dimAttr = llvm::dyn_cast<mlir::IntegerAttr>(op.getDimAttr());
      auto dim = dimAttr.getInt();
      // mlir::tosa::ReduceProdOp supports keepdim since after reducing the
      // axis, the resultant dim is reduced to 1 for example is shape is
      // (2,3,4), dim=1 then output is -> (2,1,4) So, it naturally supports
      // keepdim

      for (int64_t i = 0; i < inputShape.size(); i++) {
        if (i != dim)
          newShape.push_back(inputShape[i]);
        else {
          newShape.push_back(1);
        }
      }
      auto prod =
          mlir::RankedTensorType::get(newShape, inputType.getElementType());
      if (!keepdim) {
        auto red_prod =
            rewriter.create<mlir::tosa::ReduceProdOp>(loc, prod, input, dim);
        mlir::SmallVector<int64_t> newShape2;
        for (int64_t i = 0; i < inputShape.size(); i++) {
          if (i != dim)
            newShape2.push_back(inputShape[i]);
        }
        auto newType =
            mlir::RankedTensorType::get(newShape2, inputType.getElementType());

        auto tensorType = mlir::RankedTensorType::get(
            {static_cast<int64_t>(newShape2.size())}, rewriter.getIndexType());
        auto attr = mlir::DenseIntElementsAttr::get(
            tensorType,
            newShape2); // getting as DenseIntElementsAttr, since constShapeOp
                        // supports DenseIntElementsAttr

        // MLIRContext is the top-level object for a collection of MLIR
        // operations. It holds immortal uniqued objects like types, and the
        // tables used to unique them.

        auto types = mlir::tosa::shapeType::get(
            rewriter.getContext(),
            newShape2
                .size()); // use to construct Shape with static rank and Index
                          // element type Function signature static shapeType
                          // get(::mlir::MLIRContext *context, int rank);

        // shapeType definition in td ->
        // llvm-project/mlir/include/mlir/Dialect/Tosa/IR/TosaTypesBase.td

        auto shapeOP =
            rewriter.create<mlir::tosa::ConstShapeOp>(loc, types, attr);

        auto reshape = rewriter.create<mlir::tosa::ReshapeOp>(
            loc, newType, red_prod, shapeOP);
        rewriter.replaceOp(op, reshape);
      } else {
        auto red_prod =
            rewriter.create<mlir::tosa::ReduceProdOp>(loc, prod, input, dim);
        rewriter.replaceOp(op, red_prod);
      }
    } else { // if the dim is list of tuples
      auto arrayAttr = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(op.getDimAttr());
      for (int64_t ele : inputShape)
        newShape.push_back(ele);
      for (int64_t axis : arrayAttr.asArrayRef()) {
        newShape[axis] = 1;
        auto type =
            mlir::RankedTensorType::get(newShape, inputType.getElementType());
        input =
            rewriter.create<mlir::tosa::ReduceProdOp>(loc, type, input, axis);
      }
      if (keepdim) // If the keepdim is true
        rewriter.replaceOp(op, input);
      else {
        for (int64_t i = 0; i < arrayAttr.size(); i++) {
          newShape[arrayAttr[i]] =
              -1; // initially marking the input dim of tuples as -1
        }
        mlir::SmallVector<int64_t> newShape2;
        for (int64_t i = 0; i < inputShape.size(); i++) {
          if (newShape[i] != -1)
            newShape2.push_back(inputShape[i]); // getting the newshape
        }

        auto newType =
            mlir::RankedTensorType::get(newShape2, inputType.getElementType());

        auto tensorType = mlir::RankedTensorType::get(
            {static_cast<int64_t>(newShape2.size())},
            rewriter.getIndexType()); // for indextype

        auto attr = mlir::DenseIntElementsAttr::get(
            tensorType,
            newShape2); // getting as DenseIntElementsAttr, since constShapeOp
                        // supports only DenseIntElementsAttr

        /// Get or construct an instance of the type `Ty` with provided
        /// arguments.

        auto type =
            mlir::tosa::shapeType::get(rewriter.getContext(), newShape2.size());

        auto shapeOP =
            rewriter.create<mlir::tosa::ConstShapeOp>(loc, type, attr);

        auto reshape = rewriter.create<mlir::tosa::ReshapeOp>(loc, newType,
                                                              input, shapeOP);

        rewriter.replaceOp(op, reshape);
      }
    }
    return mlir::success();
  }
};


class EntrOpLowering : public mlir::OpRewritePattern<inductor::EntrOp> {
  using OpRewritePattern<inductor::EntrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(inductor::EntrOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc=op.getLoc();
    auto input=op.getInput();
    auto inputType=input.getType();
    auto inputTensorType = llvm::dyn_cast<mlir::RankedTensorType>(inputType);
    auto inputShape = inputTensorType.getShape();

    auto oneAttr = mlir::DenseElementsAttr::get(inputTensorType, rewriter.getFloatAttr(inputType.getElementType(), -1.0));
    auto zeroAttr = mlir::DenseElementsAttr::get(inputTensorType, rewriter.getFloatAttr(inputType.getElementType(),0.0 ));

    auto shiftType = mlir::RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto shiftAttr = mlir::DenseElementsAttr::get(shiftType, rewriter.getIntegerAttr(rewriter.getIntegerType(8), 1));
    auto constShift =rewriter.create<mlir::tosa::ConstOp>(loc, shiftType, shiftAttr);

    auto boolType=mlir::RankedTensorType::get(inputShape,  rewriter.getI1Type());

   
    auto one = rewriter.create<mlir::tosa::ConstOp>(loc,inputTensorType,oneAttr);
    auto zero = rewriter.create<mlir::tosa::ConstOp>(loc,inputTensorType,zeroAttr);

    auto posMask=rewriter.create<mlir::tosa::GreaterOp>(loc,boolType,input,zero);

    auto zeroMask=rewriter.create<mlir::tosa::EqualOp>(loc,boolType,input,zero);

    auto negMask=rewriter.create<mlir::tosa::GreaterOp>(loc,boolType,zero,input);

    // selectOp is broadcastable

    // Working of Select Op -> 1st operand:Predicate(boolean tensor), 2nd Operand:input1, 3rd Operand:input2
    // input1 tensor will be chosen when the condition(Predicate) is true.
    // input2 tensor will be chosen when the condition(Predicate) is false.

    input=rewriter.create<mlir::tosa::SelectOp>(loc,inputType,posMask,input,zero);
    // -x*log(x)
    auto negInp=rewriter.create<mlir::tosa::MulOp>(loc,inputType,input,one,constShift.getResult());
    auto res=rewriter.create<mlir::tosa::LogOp>(loc,inputType,input);
    auto interRes1=rewriter.create<mlir::tosa::MulOp>(loc,inputType,negInp,res,constShift.getResult());
    // intermediateResult2
    
    auto interRes2=rewriter.create<mlir::tosa::SelectOp>(loc,inputType,zeroMask,input,zero);
     // intermediateResult2
       
    auto interRes3=rewriter.create<mlir::tosa::SelectOp>(loc,inputType,negMask,input,zero); 
    // intermediateResult3

    auto output=rewriter.create<mlir::tosa::AddOp>(loc,inputType,interRes1,interRes2); 
    output=rewriter.create<mlir::tosa::AddOp>(loc,inputType,output,interRes3); 
    
    rewriter.replaceOp(op, output);

    return mlir::success();
}
};
class BroadcastTensorsOpLowering
    : public mlir::OpRewritePattern<inductor::BroadcastTensorsOp> {
      
      using OpRewritePattern<inductor::BroadcastTensorsOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(inductor::BroadcastTensorsOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto output = op.getResults()[0];
    auto outputType = output.getType();
    auto outputTensorType = llvm::dyn_cast<mlir::RankedTensorType>(outputType);
    auto outputShape = outputTensorType.getShape();
    mlir::SmallVector<mlir::Value> res;
    for (auto input : op.getOperands()) {
      auto inputType = input.getType();
      auto inputTensorType = llvm::dyn_cast<mlir::RankedTensorType>(inputType);
      auto inputShape = inputTensorType.getShape();
      mlir::SmallVector<int64_t> targetShape;
      if (llvm::equal(inputShape, outputShape)) {
        res.push_back(input);
        continue;
      }

      // This will pad the ones from the leftmost part according to the high
      // Rank For eg : rank1: [2,3],rank2:[1,1,3] -> OutputRank=[1,2,3]

      for (int64_t i = 0;
           i < outputTensorType.getRank() - inputTensorType.getRank(); i++)
        targetShape.push_back(1);
      for (int64_t ele :
           inputShape) // adding the remaining shape from the input shape
        targetShape.push_back(ele);

      if (!llvm::all_of(llvm::seq<int64_t>(0, outputTensorType.getRank()),
                        [&](int64_t i) {
                          return targetShape[i] == outputShape[i] ||
                                 targetShape[i] == 1;
                        })) {
        return mlir::failure();
      }
      auto newType = mlir::RankedTensorType::get(
          targetShape,
          llvm::dyn_cast<mlir::TensorType>(inputType).getElementType());
      auto tensorType = mlir::RankedTensorType::get(
          {static_cast<int64_t>(targetShape.size())}, rewriter.getIndexType());

      auto reshapedInp = rewriter.create<mlir::tosa::ReshapeOp>(
          op->getLoc(), newType, input,
          getConstShape(tensorType, targetShape, rewriter, op));
      if (llvm::equal(targetShape, outputShape)) {
        res.push_back(reshapedInp);
        continue;
      }

      auto calculateReshapeAndBroadcast = [&]() -> mlir::Value {
        mlir::SmallVector<int64_t> tileShape;
        for (int64_t i = 0; i < outputTensorType.getRank(); i++) {
          if (targetShape[i] == 1)
            tileShape.push_back(outputShape[i]);
          else
            tileShape.push_back(1);
        }

        tensorType = mlir::RankedTensorType::get(
            {static_cast<int64_t>(tileShape.size())}, rewriter.getIndexType());

        return rewriter.create<mlir::tosa::TileOp>(
            op->getLoc(), outputTensorType, reshapedInp,
            getConstShape(tensorType, tileShape, rewriter, op));
      };
      res.push_back(calculateReshapeAndBroadcast());
    }
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

class TileOpLowering : public mlir::OpRewritePattern<inductor::TileOp> {
  using OpRewritePattern<inductor::TileOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(inductor::TileOp op, mlir::PatternRewriter &rewriter) const override {
      mlir::Location loc=op.getLoc();
      auto input=op.getInput();
      auto shape=op.getShapes();
      auto inputType=input.getType();
      auto inputTensorType = llvm::dyn_cast<mlir::RankedTensorType>(inputType);
      
      mlir::SmallVector<int64_t>outputShape,tileShape;
      
      auto arrayAttr = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(op.getShapesAttr());

      for(int64_t i=0;i<inputTensorType.getRank();i++)
        outputShape.push_back(inputTensorType.getShape()[i]*arrayAttr[i]);

      for(int shape:arrayAttr.asArrayRef())
        tileShape.push_back(shape);
      
      auto tensorType = mlir::RankedTensorType::get({static_cast<int64_t>(tileShape.size())},rewriter.getIndexType());
      auto newType= mlir::RankedTensorType::get(outputShape, inputTensorType.getElementType());
      auto attr = mlir::DenseIntElementsAttr::get(tensorType, tileShape); 
      auto types = mlir::tosa::shapeType::get(rewriter.getContext(), tileShape.size());     
      auto shapeOp=rewriter.create<mlir::tosa::ConstShapeOp>(loc, types, attr);
      auto tileOp=rewriter.create<mlir::tosa::TileOp>(loc,newType,input,shapeOp);
      rewriter.replaceOp(op,tileOp);
      return mlir::success();
  }
};

class ReshapeOpLowering : public mlir::OpRewritePattern<inductor::ReshapeOp> {
  using OpRewritePattern<inductor::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(inductor::ReshapeOp op, mlir::PatternRewriter &rewriter) const override {
      mlir::Location loc=op.getLoc();
      auto input=op.getInput();
      auto shape=op.getShapes();
      auto inputType=input.getType();
      auto inputTensorType = llvm::dyn_cast<mlir::RankedTensorType>(inputType);
      auto arrayAttr = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(op.getShapesAttr());

      mlir::SmallVector<int64_t>outputShape,reshapeShape;
      
      for(int64_t i=0;i<arrayAttr.size();i++)
        outputShape.push_back(arrayAttr[i]);
      
     
      for(int shape:arrayAttr.asArrayRef())
        reshapeShape.push_back(shape);
      
      auto tensorType = mlir::RankedTensorType::get({static_cast<int64_t>(reshapeShape.size())},rewriter.getIndexType());
      auto newType= mlir::RankedTensorType::get(outputShape, inputTensorType.getElementType());

      auto attr = mlir::DenseIntElementsAttr::get(tensorType, reshapeShape); 
      auto types = mlir::tosa::shapeType::get(rewriter.getContext(), reshapeShape.size());     
      auto shapeOp=rewriter.create<mlir::tosa::ConstShapeOp>(loc, types, attr);
      
      auto reshapeOp=rewriter.create<mlir::tosa::ReshapeOp>(loc,newType,input,shapeOp);
      rewriter.replaceOp(op,reshapeOp);
      return mlir::success();
  }
};

namespace {
class InductorToTosaLowerPass
    : public mlir::PassWrapper<InductorToTosaLowerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InductorToTosaLowerPass)
  llvm::StringRef getArgument() const final {
    return "convert-inductor-to-tosa";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::TosaDialect>();
  }

  void runOnOperation() final;
};

} // namespace

void InductorToTosaLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<inductor::InductorDialect>();
  target.addLegalDialect<mlir::tosa::TosaDialect>(); // destination dialect
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<AddOpLowering>(&getContext());
  patterns.add<BatchNorm2dOpLowering>(&getContext());
  patterns.add<ProdLowering>(&getContext());
  patterns.add<BroadcastTensorsOpLowering>(&getContext());
  patterns.add<EntrOpLowering>(&getContext());
  patterns.add<TileOpLowering>(&getContext());
  patterns.add<ReshapeOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> inductor::createLowerToTosaPass() {
  return std::make_unique<InductorToTosaLowerPass>();
}