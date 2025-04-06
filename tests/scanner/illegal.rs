use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int array_1[4];
int array_2[4];

void populate_arrays(void)
{
    array_1[0] = 0;
    array_1[1] = 1;
    array_1[2] = 1;
    array_1[3] = 2;

    array_2[0] = 3;
    array_2[1] = 5;
    array_2[2] = 8;
    array_2[3] = 13;
}

int main(void)
{
    int idx, bound;

    populate_arrays();
    
    idx = 0;
    bound = 8;

    print("The first few digits of the Fibonacci sequence are:\n");
    while (idx < bound)
    {
	write(array_1[idx]);
	idx = idx + 1;
    }
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int cse141array_1[4];
int cse141array_2[4];

void cse141populate_arrays(void)
{
    cse141array_1[0] = 0;
    cse141array_1[1] = 1;
    cse141array_1[2] = 1;
    cse141array_1[3] = 2;

    cse141array_2[0] = 3;
    cse141array_2[1] = 5;
    cse141array_2[2] = 8;
    cse141array_2[3] = 13;
}

int main(void)
{
    int cse141idx, cse141bound;

    cse141populate_arrays();
    
    cse141idx = 0;
    cse141bound = 8;

    print("The first few digits of the Fibonacci sequence are:\n");
    while (cse141idx < cse141bound)
    {
	write(cse141array_1[cse141idx]);
	cse141idx = cse141idx + 1;
    }
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
