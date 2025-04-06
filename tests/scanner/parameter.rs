use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

void foo(int m,int n) {
    m = m + n;
    n = n + m;
}

int main() {
    int a;
    read(a);
    foo(a,a);
    write(a);
}


"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

void cse141foo(int cse141m,int cse141n) {
    cse141m = cse141m + cse141n;
    cse141n = cse141n + cse141m;
}

int main() {
    int cse141a;
    read(cse141a);
    cse141foo(cse141a,cse141a);
    write(cse141a);
}


"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
