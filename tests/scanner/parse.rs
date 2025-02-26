use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

void bar(void)
{
    int x, y;
    if (x > y)
    {
	return;
    }

    x = y;
    return;
}

void foo(void)
{
    bar();
}

int main(void)
{
    int x,y;
    print("Calling foo()...\n");
    foo();
    print("Called foo().\n");

    x == y;
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

void cse141bar(void)
{
    int cse141x, cse141y;
    if (cse141x > cse141y)
    {
	return;
    }

    cse141x = cse141y;
    return;
}

void cse141foo(void)
{
    cse141bar();
}

int main(void)
{
    int cse141x,cse141y;
    print("Calling foo()...\n");
    cse141foo();
    print("Called foo().\n");

    cse141x == cse141y;
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
