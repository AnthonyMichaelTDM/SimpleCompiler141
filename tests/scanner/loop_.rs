use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int main() {
    int a, sum;
    read(a);
    sum = 0;
    while (a>0) {
        sum = sum + a;
        a = a - 1;
    }
    write(sum);
}

"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int main() {
    int cse141a, cse141sum;
    read(cse141a);
    cse141sum = 0;
    while (cse141a>0) {
        cse141sum = cse141sum + cse141a;
        cse141a = cse141a - 1;
    }
    write(cse141sum);
}

"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
