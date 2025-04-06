use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

void foo( ) {
    int a;
    read(a) ;
    write(a) ;
}

int main( ) {
  foo( ) ;
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

void cse141foo( ) {
    int cse141a;
    read(cse141a) ;
    write(cse141a) ;
}

int main( ) {
  cse141foo( ) ;
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
