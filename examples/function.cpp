
#include "nabo/function.h"

int main() 
{

	using namespace Nabo;
	using namespace Eigen;

    Function21f F1, F2;
    for (double x = -10; x <= 10; x += 0.2) {
        for (double y = -10; y <= 10; y += 0.2) {
            double d = hypot(x,y);
            if (d > 1e-6) {
                F1.set(x,y,sin(d) / d);
            } else {
                F1.set(x,y,1.0);
            }
            F2.set(x,y,(x + y) / 20);
        }
    }
    F1.compile();
    F2.compile();

    F1.print("F1");
    F2.print("F2");

    Function21f F1plusF2 = F1 + F2; F1plusF2.print("F1plusF2");
    Function21f F1minusF2 = F1 - F2;F1minusF2.print("F1minusF2");
    Function21f F1mulF2 = F1 * F2;F1mulF2.print("F1mulF2");
    Function21f F1plus1 = F1 + 1.0; F1plus1.print("F1plus1");
    Function21f F2divF1 = F2 / (F1 + 2.0);F2divF1.print("F2divF1");
    Function21f expF1 = F1.map(exp);expF1.print("expF1");


    return 0;
}

