/* Mock with missing symbols */
#include <string.h>

void some_other_function(float* x) {
    memset(x, 0, 64 * sizeof(float));
}
