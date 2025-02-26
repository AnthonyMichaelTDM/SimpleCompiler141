use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int array[16];

void initialize_array(void)
{
    int idx, bound;
    bound = 16;

    idx = 0;
    while (idx < bound)
    {
	array[idx] = -1;
	idx = idx + 1;
    }
}

int fib(int val)
{
    if (val < 2)
    {
	return 1;
    }
    if (array[val] == -1)
    {
	array[val] = fib(val - 1) + fib(val - 2);
    }

    return array[val];
}

int main(void)
{
    int idx, bound;
    bound = 16;

    initialize_array();
    
    idx = 0;

    print("The first few digits of the Fibonacci sequence are:\n");
    while (idx < bound)
    {
	write(fib(idx));
	idx = idx + 1;
    }
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int cse141array[16];

void cse141initialize_array(void)
{
    int cse141idx, cse141bound;
    cse141bound = 16;

    cse141idx = 0;
    while (cse141idx < cse141bound)
    {
	cse141array[cse141idx] = -1;
	cse141idx = cse141idx + 1;
    }
}

int cse141fib(int cse141val)
{
    if (cse141val < 2)
    {
	return 1;
    }
    if (cse141array[cse141val] == -1)
    {
	cse141array[cse141val] = cse141fib(cse141val - 1) + cse141fib(cse141val - 2);
    }

    return cse141array[cse141val];
}

int main(void)
{
    int cse141idx, cse141bound;
    cse141bound = 16;

    cse141initialize_array();
    
    cse141idx = 0;

    print("The first few digits of the Fibonacci sequence are:\n");
    while (cse141idx < cse141bound)
    {
	write(cse141fib(cse141idx));
	cse141idx = cse141idx + 1;
    }
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
