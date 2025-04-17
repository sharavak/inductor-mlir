#ifndef INDUCTOR_DIALECT
#define INDUCTOR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"



def Inductor_Dialect : Dialect{
    let name="inductor"; // name of the dialect
    let summary ="A Inductor dialect";
    let description =[{
        Just a simple inductor dialect
    }];

    let cppNamespace="::inductor";
}

class Inductor_Op<string mnemonic, list<Trait> traits = []> :Op<Inductor_Dialect, mnemonic, traits>;


#endif // INDUCTOR_DIALECT