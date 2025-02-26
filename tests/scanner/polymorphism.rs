use pretty_assertions::assert_str_eq;
use super::scan;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define binary int
#define decimal int

void print_two(int a, int b) {  
    write(a);
    write(b);
}

int main() {
    binary b;
    decimal a;
    read(a);
    read(b);  
    print_two(a, b);
    print_two(b, a);
}

"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define binary int
#define decimal int

void cse141print_two(int cse141a, int cse141b) {  
    write(cse141a);
    write(cse141b);
}

int main() {
    binary cse141b;
    decimal cse141a;
    read(cse141a);
    read(cse141b);  
    cse141print_two(cse141a, cse141b);
    cse141print_two(cse141b, cse141a);
}

"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
