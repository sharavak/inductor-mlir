#include "Utils/ShapeType.h"

// used to get the shape type with creating tosa::ConstShapeOp
mlir::Value getConstShape(mlir::RankedTensorType tensorType,
                          mlir::SmallVector<int64_t> &targetShape,
                          mlir::PatternRewriter &rewriter,
                          inductor::BroadcastTensorsOp op) {
  mlir::DenseIntElementsAttr attr =
      mlir::DenseIntElementsAttr::get(tensorType, targetShape);
  mlir::Type type =
      mlir::tosa::shapeType::get(rewriter.getContext(), targetShape.size());
  return rewriter.create<mlir::tosa::ConstShapeOp>(op->getLoc(), type, attr);
}
