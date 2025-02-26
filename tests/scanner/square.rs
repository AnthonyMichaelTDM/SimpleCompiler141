use pretty_assertions::assert_str_eq;
use super::scan;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int square(int x)
{
    int x;
    x = 10;
    return x * x;
}

int main(void)
{
    int val;
    print("Give me a number: ");
    read(val);

    print("Your number squared is: ");
    write(square(val));
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int cse141square(int cse141x)
{
    int cse141x;
    cse141x = 10;
    return cse141x * cse141x;
}

int main(void)
{
    int cse141val;
    print("Give me a number: ");
    read(cse141val);

    print("Your number squared is: ");
    write(cse141square(cse141val));
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
