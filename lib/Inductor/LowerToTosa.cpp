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
    rewriter.replaceOp(op.getOperation(), addop.getOperation());
    return mlir::success();
}
};

void change( mlir::SmallVector<int64_t> shape,auto inputShape){
  
}

class BatchNorm2dOpLowering : public mlir::OpRewritePattern<inductor::BatchNorm2dOp> {
  using OpRewritePattern<inductor::BatchNorm2dOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(inductor::BatchNorm2dOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = input.getType(); 
    auto inputTensorType = inputType.cast<mlir::RankedTensorType>();
    auto inputShape = inputTensorType.getShape();
  
    mlir::SmallVector<int64_t> shape;
    shape.push_back(1);
    for(int64_t i=1;i<inputShape.size();i++){
      shape.push_back(inputShape[i]);
    }      
    
    // for mean sum
    auto ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value btsum = rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, input, rewriter.getI32IntegerAttr({0}));
    shape.clear();
    shape.push_back(1);
    shape.push_back(ip.getShape()[1]);
    shape.push_back(1);
    shape.push_back(ip.getShape()[3]);
        
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value wisum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, btsum, rewriter.getI32IntegerAttr({2}));
    shape.clear();
    shape.push_back(1);
    shape.push_back(ip.getShape()[1]);
    shape.push_back(1);
    shape.push_back(1);

    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value hisum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, wisum, rewriter.getI32IntegerAttr({3}));
    float val=1.0/(inputShape[0] * inputShape[2] * inputShape[3]);
    auto const_type = mlir::RankedTensorType::get({1,inputShape[1],1,1}, rewriter.getF32Type());
    auto const_attr = mlir::DenseElementsAttr::get(const_type, val);
    auto divisor = rewriter.create<mlir::tosa::ConstOp>(loc,const_type,const_attr);

    //https://github.com/llvm/torch-mlir/blob/596b58ea243266996331689d50f59044a01eb367/lib/Conversion/TorchToTosa/TosaLegalizeUtils.cpp#L153
    
                                            //  shape   
    auto shiftType = mlir::RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto shiftAttr = mlir::DenseElementsAttr::get(shiftType, rewriter.getIntegerAttr(rewriter.getIntegerType(8), 1));
    auto constShift =rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), shiftType, shiftAttr);
    auto width_avg = rewriter.create<mlir::tosa::MulOp>(loc,ip,hisum,divisor,constShift.getResult());
    auto var_sub=rewriter.create<mlir::tosa::SubOp>(loc,inputType,input,width_avg);
  

    // for variance sum
    shape.clear();
    shape.push_back(1);
    for(int64_t i=1;i<inputShape.size();i++){
      shape.push_back(inputShape[i]);
    }      
 

    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    auto btvarsum = rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, var_sub, rewriter.getI32IntegerAttr({0}));
    shape.clear();
    shape.push_back(1);
    shape.push_back(ip.getShape()[1]);
    shape.push_back(1);
    shape.push_back(ip.getShape()[3]);
        
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    auto wivarsum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, btvarsum, rewriter.getI32IntegerAttr({2}));
    shape.clear();
    shape.push_back(1);
    shape.push_back(ip.getShape()[1]);
    shape.push_back(1);
    shape.push_back(1);
    ip = mlir::RankedTensorType::get(shape, inputType.getElementType());
    mlir::Value b_sum= rewriter.create<mlir::tosa::ReduceSumOp>(loc, ip, wivarsum, rewriter.getI32IntegerAttr({3}));
    
    auto batch_var = rewriter.create<mlir::tosa::MulOp>(loc,inputType,b_sum,divisor,constShift.getResult());
    float eps=1e-5;
    auto eps_type = mlir::RankedTensorType::get({1,1,1,1}, rewriter.getF32Type());
    auto epsAttr = mlir::DenseElementsAttr::get(eps_type, rewriter.getFloatAttr(rewriter.getF32Type(), eps));
    auto epsinter =rewriter.create<mlir::tosa::ConstOp>(op->getLoc(), eps_type, epsAttr);
    auto batch_v=rewriter.create<mlir::tosa::AddOp>(loc,inputType,batch_var,epsinter);
    auto rec_sqrt=rewriter.create<mlir::tosa::RsqrtOp>(loc,inputType,batch_v);
    mlir::tosa::MulOp xnorm=rewriter.create<mlir::tosa::MulOp>(loc,inputType,b_sum,rec_sqrt,constShift.getResult());
    rewriter.replaceOp(op, xnorm);
                  
    /*llvm::outs()<<inputShape.size()<<"\n";
    llvm::outs()<< inputTensorType<<"\n";
    llvm::outs()<< rewriter.getI32IntegerAttr(0)<<"\n"
    llvm::outs()<<batch_var<<"\n"<<"batch_var"<<"\n";

    */
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
  target.addIllegalOp<inductor::AddOp>();
  target.addIllegalOp<inductor::BatchNorm2dOp>();
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
