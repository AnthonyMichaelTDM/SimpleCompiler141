use pretty_assertions::assert_str_eq;
use super::scan;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

void bar@(void)
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
    bar@();
}

int main(void)
{
    foo();
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

void cse141barThe input program contains errors for scanning."#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
