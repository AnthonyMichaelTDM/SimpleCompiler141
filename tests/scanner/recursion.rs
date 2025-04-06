use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int recursionsum(int n) {
    if (n==0) {
        return 0;
    }
    return n + recursionsum(n-1);
}

int main() {
    int a;
    read(a);
    write(recursionsum(a));
}


"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int cse141recursionsum(int cse141n) {
    if (cse141n==0) {
        return 0;
    }
    return cse141n + cse141recursionsum(cse141n-1);
}

int main() {
    int cse141a;
    read(cse141a);
    write(cse141recursionsum(cse141a));
}


"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
