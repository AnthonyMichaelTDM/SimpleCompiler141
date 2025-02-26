use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int main() {
    int a, b;
    read(a);
    read(b);
    if (a>=b) {
        write(a);
    }
    if (b>a) {
        write(b);
    }
}

"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int main() {
    int cse141a, cse141b;
    read(cse141a);
    read(cse141b);
    if (cse141a>=cse141b) {
        write(cse141a);
    }
    if (cse141b>cse141a) {
        write(cse141b);
    }
}

"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
