use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int max(int a, int b) {
    if (a>b) {
        return a;
    }
    return b;
}

int main() {
    int a,b;
    read(a);
    read(b);

    write(max(a,b));
}

"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int cse141max(int cse141a, int cse141b) {
    if (cse141a>cse141b) {
        return cse141a;
    }
    return cse141b;
}

int main() {
    int cse141a,cse141b;
    read(cse141a);
    read(cse141b);

    write(cse141max(cse141a,cse141b));
}

"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
