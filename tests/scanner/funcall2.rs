use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int c()
{
    return 1;
}

int b()
{
    return 2;
}

int a()
{
    return 3;
}

int foo(int a, int b, int c)
{
    return (a*3 + b*2 + c);
}

int main() 
{
    int val;
    val = foo(a(), b(), c());

    print("I calculate the answer to be: ");
    write(val);
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int cse141c()
{
    return 1;
}

int cse141b()
{
    return 2;
}

int cse141a()
{
    return 3;
}

int cse141foo(int cse141a, int cse141b, int cse141c)
{
    return (cse141a*3 + cse141b*2 + cse141c);
}

int main() 
{
    int cse141val;
    cse141val = cse141foo(cse141a(), cse141b(), cse141c());

    print("I calculate the answer to be: ");
    write(cse141val);
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
