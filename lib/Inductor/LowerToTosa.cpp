#include "Inductor/InductorDialect.h"
#include "Inductor/InductorOps.h"
#include "Inductor/InductorPasses.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"


#include<iostream>
using namespace std;


class AddOpLowering : public mlir::OpRewritePattern<inductor::AddOp> {
  using OpRewritePattern<inductor::AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(inductor::AddOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Type resultType = op.getResult().getType();
    mlir::tosa::AddOp addop=rewriter.create<mlir::tosa::AddOp>(op.getLoc(),resultType, op.getOperands());
    rewriter.replaceOp(op, addop);
    return mlir::success();
}
};

class BatchNorm2dOpLowering : public mlir::OpRewritePattern<inductor::BatchNorm2dOp> {

  using OpRewritePattern<inductor::BatchNorm2dOp>::OpRewritePattern;
  
  mlir::LogicalResult matchAndRewrite(inductor::BatchNorm2dOp op,
                                      mlir::PatternRewriter &rewriter) const override {

    // General Formula -> (x-mean)/sqrt(variance+eps)
    // Variance Formula -> (x-mean)*(x-mean)/N
    // Average(Mean) Formula -> (sum of all elements)/N

    mlir::Location loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = input.getType(); 
    auto inputTensorType = inputType.cast<mlir::RankedTensorType>();
    auto inputShape = inputTensorType.getShape();
    auto affine=op.getAffine();

    mlir::SmallVector<int64_t> shape(4,0);
    shape={1,inputShape[1],inputShape[2],inputShape[3]};
    // ---- For getting avg sum for the axis 0 2 3 

    auto ip = mlir::RankedTensorType::get(shape, inputType.getElementType());// (1,C,W,H)
    mlir::Value btsum = rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, input, rewriter.getI32IntegerAttr({0}));
    shape={1,ip.getShape()[1],1,ip.getShape()[3]};

    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value wisum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, btsum, rewriter.getI32IntegerAttr({2}));// (1,C,1,H)
    shape={1,ip.getShape()[1],1,1};

    ip = mlir::RankedTensorType::get(shape, inputType.getElementType()); // (1,C,1,1)
    mlir::Value hisum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, wisum, rewriter.getI32IntegerAttr({3}));// batch sum (1,C,1,1)
    
    float val=1.0/(inputShape[0] * inputShape[2] * inputShape[3]);
    auto const_type = mlir::RankedTensorType::get({1,inputShape[1],1,1}, rewriter.getF32Type());
    auto const_attr = mlir::DenseElementsAttr::get(const_type, val);
    
    auto divisor = rewriter.create<mlir::tosa::ConstOp>(loc,const_type,const_attr); // (1,C,1,1)

    auto shiftType = mlir::RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto shiftAttr = mlir::DenseElementsAttr::get(shiftType, rewriter.getIntegerAttr(rewriter.getIntegerType(8), 1));
    auto constShift =rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), shiftType, shiftAttr);
    
    auto width_mean = rewriter.create<mlir::tosa::MulOp>(loc,ip,hisum,divisor,constShift.getResult());// batch avg (1,C,1,1)

    // Variance sum for the axis 0 2 3
    // (X-mean)
    auto var_sub=rewriter.create<mlir::tosa::SubOp>(loc,inputType,input,width_mean);
    // {b,h,w,h}    

    shape={1,inputShape[1],inputShape[2],inputShape[3]};
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    auto btVarSum = rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, var_sub, rewriter.getI32IntegerAttr({0}));
    
    shape={1,ip.getShape()[1],1,ip.getShape()[3]};
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    auto wiVarSum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, btVarSum, rewriter.getI32IntegerAttr({2}));
    
    shape={1,ip.getShape()[1],1,1};
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value bt_sum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, wiVarSum, rewriter.getI32IntegerAttr({3}));// variance sum
    
    auto batch_var = rewriter.create<mlir::tosa::MulOp>(loc,inputType,bt_sum,divisor,constShift.getResult());    // variance                                    

    float eps=1e-5;
    auto epsType = mlir::RankedTensorType::get({1,1,1,1}, rewriter.getF32Type());
    auto epsAttr = mlir::DenseElementsAttr::get(epsType, rewriter.getFloatAttr(rewriter.getF32Type(), eps));
    auto epsinter =rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), epsType, epsAttr);

    auto varAdd=rewriter.create<mlir::tosa::AddOp>(loc,inputType,batch_var,epsinter); // (variance+eps)

    auto rec_sqrt=rewriter.create<mlir::tosa::RsqrtOp>(loc,inputType,varAdd); // 1/(sqrt(variance+eps))

    auto  xnorm=rewriter.create<mlir::tosa::MulOp>(loc,inputType,bt_sum,rec_sqrt,constShift.getResult()); 
    if(affine){
      // #x_norm = x_norm*gamma + beta
      auto gammaType=mlir::RankedTensorType::get({1,inputShape[1],1,1}, rewriter.getF32Type() );
      auto gammaAttr = mlir::DenseElementsAttr::get(gammaType, rewriter.getFloatAttr(rewriter.getF32Type(), 1.0));
      auto gamma = rewriter.create<mlir::tosa::ConstOp>(loc,gammaType,gammaAttr); // (1,C,1,1)
      xnorm = rewriter.create<mlir::tosa::MulOp>(loc,inputType,xnorm,gamma,constShift.getResult()); 

      auto betaType=mlir::RankedTensorType::get({1,inputShape[1],1,1}, rewriter.getF32Type());
      auto betaAttr = mlir::DenseElementsAttr::get(betaType, rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
      auto beta = rewriter.create<mlir::tosa::ConstOp>(loc,betaType,betaAttr); // (1,C,1,1)

      auto new_xnorm=rewriter.create<mlir::tosa::AddOp>(loc,inputType,xnorm,beta);
      rewriter.replaceOp(op, new_xnorm);
    }else{
      rewriter.replaceOp(op, xnorm);
    }
                  
    return mlir::success();
  }
};

namespace 
{
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

} 


void InductorToTosaLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<inductor::InductorDialect>();
  target.addLegalDialect<mlir::tosa::TosaDialect>(); // destination dialect
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<AddOpLowering>(&getContext());
  patterns.add<BatchNorm2dOpLowering>(&getContext());


  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,std::move(patterns)))) {
      signalPassFailure();
    }
}
  
std::unique_ptr<mlir::Pass> inductor::createLowerToTosaPass() {
    return std::make_unique<InductorToTosaLowerPass>();
  }